import pandas as pd 
import numpy as np
import torch 
from torch import Tensor
from einops import rearrange

from tsl.utils.casting import torch_to_numpy
from tsl.metrics import numpy as numpy_metrics

from my_tsl.imputation import prediction_dataframe


def format_output(output):
    # baseline models return a dict
    if isinstance(output, dict):
        y_hat = [[output['imputed_data']], []]
        y_hat_loss = output.copy()
                                    # pinall components   # additional loss terms
    # some models returns a list: [(imp1, imp2, imp3),    (pred1, pred2, ...)]
    elif isinstance(output, list):
            for i in range(len(output)):
                if not isinstance(output[i], list) and not isinstance(output[i], tuple):
                    output[i] = [output[i]]
            y_hat      = output.copy()
            y_hat_loss = output.copy()

    elif isinstance(output, Tensor):
        # other tsl models that returns a single tensor
        y_hat      = [[output], []]
        y_hat_loss = [[output], []]
    
    else:
        raise ValueError('format_output: invalid output type')
    
    return y_hat, y_hat_loss


def unpack_output(y_hat, pinball):
    if len(y_hat) == 3 and pinball:
        lower, median, upper = y_hat      
    elif len(y_hat) == 1:
        lower, median, upper = None, y_hat[0], None 
    else:
        raise ValueError('unpack_output: invalid number of returned predictions')
    return lower, median, upper


def get_deltas(mask):
        """Calculate time delta calculated as in Che et al, 2018, see
        https://www.nature.com/articles/s41598-018-24271-9.
        Adapted from the torchtime package, see
        https://philipdarke.com/torchtime/index.html"""
        time_mask = mask.clone().transpose(1, 2)
        time_stamp = torch.arange(time_mask.size()[2], device=mask.device).float()
        time_stamp = time_stamp.repeat(time_mask.size()[0], time_mask.size()[1], 1)

        time_delta = time_stamp.clone()
        time_delta[:, :, 0] = 0
        time_mask[:, :, 0] = 1
        # Time of previous observation if data missing
        time_delta = time_delta.gather(-1, torch.cummax(time_mask, -1)[1])
        # Calculate time delta
        time_delta = torch.cat(
            (
                time_delta[:, :, 0].unsqueeze(2),  # t = 0
                time_stamp[:, :, 1:]
                - time_delta[:, :, :-1],  # i.e. time minus time of previous observation
            ),
            dim=2,
        )
        return time_delta.transpose(1, 2)


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

def get_consistency_loss(pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean()
        return loss

def baselines_loss(results, 
                   y_loss,
                   mask):
    """process results and losses for each training step"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results["total_loss"] = torch.tensor(0.0, device=device)

    # preprare
    if torch.isnan(y_loss).any():
        y_loss = torch.where(torch.isnan(y_loss), torch.zeros_like(y_loss), y_loss)
    y_loss = rearrange(y_loss, 'b t n f -> (b n) t f')   
    mask = rearrange(mask, 'b t n f -> (b n) t f')

    # BRITS
    if results['model_name']=='BRITS':
        consistency_loss_weight = 0.1
        reconstruction_loss_weight = 1
        # consistency loss
        results["total_loss"] += get_consistency_loss(results["imputed_data_fwd"], results["imputed_data_bwd"]) * consistency_loss_weight
        # other terms
        reconstruction_loss_fwd = masked_mae_cal(results['first_fwd'], y_loss, mask) + \
                                  masked_mae_cal(results['second_fwd'], y_loss, mask) + \
                                  masked_mae_cal(results['third_fwd'], y_loss, mask)
        reconstruction_loss_bwd = masked_mae_cal(results['first_bwd'], y_loss, mask) + \
                                  masked_mae_cal(results['second_bwd'], y_loss, mask) + \
                                  masked_mae_cal(results['third_bwd'], y_loss, mask)
        results["total_loss"] += (reconstruction_loss_fwd + reconstruction_loss_bwd) * reconstruction_loss_weight

    # SAITS
    elif results["model_name"]=="SAITS":
        results["total_loss"] += (masked_mae_cal(results['first'], y_loss, mask) + \
                                  masked_mae_cal(results['second'], y_loss, mask) + \
                                  masked_mae_cal(results['third'], y_loss, mask)) / 3.
    return results


def collect_one_result(out):
    out = torch_to_numpy(out)
    y_hat, y_true = out['y_hat'], out['y']
    mask = out.get('mask', None)
    val_mask = out.get('val_mask', None)
    test_mask = out.get('test_mask', None)
    
    res = dict(train_mae=numpy_metrics.mae(y_hat, y_true, mask),
               train_mre=numpy_metrics.mre(y_hat, y_true, mask),
               train_mape=numpy_metrics.mape(y_hat, y_true, mask))
    
    res.update(dict(val_mae=numpy_metrics.mae(y_hat, y_true, val_mask),
                    val_rmse=numpy_metrics.rmse(y_hat, y_true, val_mask),
                    val_mape=numpy_metrics.mape(y_hat, y_true, val_mask)))
    
    res.update(dict(test_mae=numpy_metrics.mae(y_hat, y_true, test_mask),
                    test_rmse=numpy_metrics.rmse(y_hat, y_true, test_mask),
                    test_mape=numpy_metrics.mape(y_hat, y_true, test_mask)))
    return res



def collect_results(out_train, out_val, out_test):
    # train
    output_train = torch_to_numpy(out_train)
    y_hat_train, y_true_train, mask_train = output_train['y_hat'], \
                                            output_train['y'], \
                                            output_train.get('mask', None)
    res = dict(train_mae=numpy_metrics.mae(y_hat_train, y_true_train, mask_train),
               train_mre=numpy_metrics.mre(y_hat_train, y_true_train, mask_train),
               train_mape=numpy_metrics.mape(y_hat_train, y_true_train, mask_train))
    
    # val
    output_val = torch_to_numpy(out_val)
    y_hat_val, y_true_val, mask_val = output_val['y_hat'], \
                                      output_val['y'], \
                                      output_val.get('mask', None)
    res.update(dict(val_mae=numpy_metrics.mae(y_hat_val, y_true_val, mask_val),
                    val_rmse=numpy_metrics.rmse(y_hat_val, y_true_val, mask_val),
                    val_mape=numpy_metrics.mape(y_hat_val, y_true_val, mask_val)))
    
    # test
    output_test = torch_to_numpy(out_test)
    y_hat_test, y_true_test, mask_test = output_test['y_hat'], \
                                         output_test['y'], \
                                         output_test.get('mask', None)
    res.update(dict(test_mae=numpy_metrics.mae(y_hat_test, y_true_test, mask_test),
                    test_rmse=numpy_metrics.rmse(y_hat_test, y_true_test, mask_test),
                    test_mape=numpy_metrics.mape(y_hat_test, y_true_test, mask_test))) 

    return res

    


def aggregate_one_prediction(out, dm):
    window=dm.window
    index = dm.dataframe().index
    columns = dm.dataframe().columns
    
    sl = dm.train_slice # = dm.val_slice = dm.test_slice
    index = [index[i:i+window] for i in range(sl[0], sl[-1]-window+2)]
    df_hat = prediction_dataframe(out, index=index, columns=columns, aggregate_by='mean')
    return df_hat, sl



def aggregate_predictions(out_train, out_val, out_test, dm):
    window=dm.window
    index = dm.dataframe().index
    columns = dm.dataframe().columns
    
    # train
    tr_sl = dm.train_slice
    tr_index = [index[i:i+window] for i in range(tr_sl[0], tr_sl[-1]-window+2)]
    df_hat_train = prediction_dataframe(out_train['y_hat'], index=tr_index, columns=columns, aggregate_by='mean')
    df_true_train = prediction_dataframe(out_train['y'], index=tr_index, columns=columns, aggregate_by='mean')
    df_mask_train = prediction_dataframe(out_train['mask'], index=tr_index, columns=columns, aggregate_by='last')
    
    separator1 = pd.DataFrame(data=np.nan, index=[index[(tr_sl[-1]+1).tolist()]], columns=columns)
    
    # val
    va_sl = dm.val_slice
    va_index = [index[i:i+window] for i in range(va_sl[0], va_sl[-1]-window+2)]
    df_hat_val = prediction_dataframe(out_val['y_hat'], index=va_index, columns=columns, aggregate_by='mean')
    df_true_val = prediction_dataframe(out_val['y'], index=va_index, columns=columns, aggregate_by='mean')
    df_mask_val = prediction_dataframe(out_val['mask'], index=va_index, columns=columns, aggregate_by='last')
    
    separator2 = pd.DataFrame(data=np.nan, index=[index[(va_sl[-1]+1).tolist()]], columns=columns)
    
    # test
    te_sl = dm.test_slice
    te_index = [index[i:i+window] for i in range(te_sl[0], te_sl[-1]-window+2)]
    df_hat_test = prediction_dataframe(out_test['y_hat'], index=te_index, columns=columns, aggregate_by='mean')
    df_true_test = prediction_dataframe(out_test['y'], index=te_index, columns=columns, aggregate_by='mean')
    df_mask_test = prediction_dataframe(out_test['mask'], index=te_index, columns=columns, aggregate_by='last')
    
    # aggregate
    df_hat = pd.concat([df_hat_train, separator1, df_hat_val, separator2, df_hat_test])
    df_true = pd.concat([df_true_train, separator1, df_true_val, separator2, df_true_test])
    df_mask = pd.concat([df_mask_train, separator1, df_mask_val, separator2, df_mask_test])
    
    # return df_hat, [tr_sl, va_sl, te_sl]
    return df_hat, df_true, df_mask, [tr_sl, va_sl, te_sl]


