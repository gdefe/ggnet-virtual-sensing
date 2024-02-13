# general imports
import os
import wandb
import numpy as np
import torch
import pickle
from omegaconf import DictConfig
import random

# lightning imports
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# tsl imports
from tsl import logger
from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.data.datamodule.splitters import CustomSplitter
from tsl.experiment import Experiment
from tsl.metrics import torch as torch_metrics
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy

# my modifications to tsl
# datasets
from virtual_sensing.datasets.climate_capitals import ClimateCapitals
from virtual_sensing.datasets.photovoltaic import Photovoltaic
# mask and splitter
from virtual_sensing.imputation import add_virtual_sensors
from virtual_sensing.imputation_dataset import ImputationDataset
from virtual_sensing.nodewise_splitter import nodewise_index
# baselines
from virtual_sensing.imputer import Imputer
from tsl.nn.models.temporal.rnn_imputers_models import RNNImputerModel
from virtual_sensing.models.baselines.birnn import BiRNNImputerModel
from virtual_sensing.models.baselines.rnnemb import RNNImputerEmbModel
from virtual_sensing.models.baselines.birnnemb import BiRNNImputerEmbModel
from virtual_sensing.models.baselines.time_then_space import TimeThenSpaceModel
from virtual_sensing.models.baselines.brits import BRITSModel
from virtual_sensing.models.baselines.saits import SAITSModel
from tsl.nn.models.stgn.grin_model import GRINModel
from virtual_sensing.models.ggnet import GGNetModel

# metrics
from virtual_sensing.my_metrics import MultivariateMaskedMRE, MaskedVRE, MaskedPinballLoss

# utils imports 
from torch_geometric.utils import to_dense_adj
from utils import aggregate_one_prediction


# given the model name, return the tsl model class
def get_model_class(model_str):
    if model_str == 'rnn':
        model = RNNImputerModel
    elif model_str == 'birnn':
        model = BiRNNImputerModel
    elif model_str == 'rnn_emb':
        model = RNNImputerEmbModel
    elif model_str == 'birnn_emb':
        model = BiRNNImputerEmbModel
    elif model_str == 'time_then_space':
        model = TimeThenSpaceModel
    elif model_str == 'brits':
        model = BRITSModel
    elif model_str == 'saits':
        model = SAITSModel
    elif model_str == 'grin':
        model = GRINModel
    elif model_str == 'ggnet':
        model = GGNetModel
    else:
        raise ValueError(f"Model {model_str} not available.")
    return model


# given the dataset name and various settings, load the dataset and generate the masks for the virtual sensors 
def get_dataset(dataset_name: str,
                root_dir=None,  # where to find the dataset, e.g. 'data/NASA_data'
                p_fault=0.,     # prob. of masking a window of steps
                p_noise=0.,     # prob. of masking one step
                noise_seed=12345,   # seed for random missing values generation
                val_size=0.1,       # % of validation sensors
                test_size=0.2,      # % of testing sensors
                custom_vs_train=[], # list of pairs, those nodes/channel pairs will be in the train
                custom_vs_val=[],   # list of pairs, those nodes/channel pairs will be in the val
                custom_vs_test=[],  # list of pairs, those nodes/channel pairs will be in the test
                remove_data_frac=0.0,  # completely remove some data, for robustness testing (climate only)
                remove_vs_frac=0.0,    # completely remove some channels, for robustness testing (climate only)
                sampling_seed=1,       # different sampling of random locations over US (photovoltaic only)
                ):  

    # load the dataset, maybe normalize and handle missing values
    if dataset_name in ['clmDaily', 'clmHourly']:
        loaded = ClimateCapitals(name=dataset_name,
                                 root=root_dir,
                                 impute_zeros=False,  # we leav nans into the data, this assure correct handlnig during model forwards
                                 normalize=False,     # no channel-wise standardization at this step
                                 remove_data_frac=remove_data_frac,
                                 remove_vs_frac=remove_vs_frac)
        
    elif 'photovoltaic' in dataset_name:
        npoints = int(''.join(filter(str.isdigit, dataset_name)))
        loaded = Photovoltaic(root=root_dir,
                              npoints=npoints,
                              sampling_seed=sampling_seed,
                              impute_zeros=False,  # we leav nans into the data, this assure correct handlnig during model forwards
                              normalize=False)     # no channel-wise standardization at this step
        
        # sample validation and test channels from the photovoltaic output variable only
        custom_vs_test = random.sample(list(range(npoints)), int(npoints * test_size))       
        custom_vs_val = random.sample(list(set(range(npoints)) - set(custom_vs_test)), int(npoints * val_size))
        custom_test = [(i, 10) for i in custom_vs_test]
        custom_val = [(i, 10) for i in custom_vs_val]

    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    
    # compute evaluation mask (i.e. training, val and test channels)
    loaded_and_masked = add_virtual_sensors(dataset=loaded,
                                            p_fault=p_fault,
                                            p_noise=p_noise,
                                            noise_seed=noise_seed, # seed for noise mask
                                            val_size=val_size,
                                            test_size=test_size,
                                            custom_vs_train=custom_vs_train,
                                            custom_vs_val=custom_vs_val,
                                            custom_vs_test=custom_vs_test,
                                            custom_val=custom_val if 'photovoltaic' in dataset_name else [],
                                            custom_test=custom_test if 'photovoltaic' in dataset_name else [],
                                            min_seq=12, max_seq=12 * 4)

    return loaded_and_masked
    


def run_train(cfg: DictConfig):
    # display where log, checkpoint and configs backup will be saved
    logger.info(f'FILENAME: {cfg.save_dir}')

    # save a copy of the config file for easy access in evaluation
    pickle.dump(cfg, open(f'{cfg.save_dir}/config.pkl', 'wb'))

    # set the seed
    if cfg.seed < 0: cfg.seed = np.random.randint(1e9)
    seed_everything(cfg.seed)
    logger.info(f'SEED: {cfg.seed}')

    ########################################
    # data module                          #
    ########################################
    # auxiliary variable: if there are some custom virtual sensors, always get inside the 'if' statement and recreate the dataset and masks
    custom=cfg.get('custom')
    if cfg.get('custom_vs_train') or cfg.get('custom_vs_val') or cfg.get('custom_vs_test') or cfg.get('custom_test'):
        custom = True
    if cfg.get('remove_data_frac') > 0 or cfg.get('remove_vs_frac') > 0:
        custom = True
    # if datamodule was never created with this settings and thus does not exist in the folder, create and save it
    # if was already created with this settings, then load it directly from folder
    if not f'{cfg.datamodule_name}.pth' in os.listdir('datamodules') or custom:
        logger.info(f'CREATING DATAMODULE: {cfg.datamodule_name}.pth')
        # type MissingValuesMixin
        # dataset.mask          --->  True = Valid              // False = Invalid
        # dataset.training_mask --->  True = Valid and unmasked // False = Invalid or masked        (mask & ~eval_mask)
        # dataset.eval_mask     --->  True = Valid and masked   // False = Invalid or unmasked      (mask & sampled_faults & all_VS)
        # dataset.val_mask      --->  True = Valid and val      // False = Invalid or train or test (mask & sampled_faults & val_VS)
        # dataset.test_mask     --->  True = Valid and test     // False = Invalid or train or val  (mask & sampled_faults & test_VS)
        dataset = get_dataset(dataset_name=cfg.dataset.name,
                              root_dir=cfg.dataset.root_dir,
                              p_fault=cfg.get('p_fault'),
                              p_noise=cfg.get('p_noise'),
                              noise_seed=cfg.seed,
                              val_size=cfg.get('val_size'),
                              test_size=cfg.get('test_size'),
                              custom_vs_train=cfg.get('custom_vs_train'),
                              custom_vs_val=cfg.get('custom_vs_val'),
                              custom_vs_test=cfg.get('custom_vs_test'),
                              sampling_seed=cfg.get('sampling_seed') if 'photovoltaic' in cfg.dataset.name else None,
                              remove_data_frac=cfg.get('remove_data_frac'),
                              remove_vs_frac=cfg.get('remove_vs_frac'))

        # encode time of the day and use it as exogenous variable
        # not used for now
        # covariates = {'u': dataset.datetime_encoded('day').values}

        # get adjacency matrix
        if cfg.dataset.connectivity.method:
            adj = dataset.get_connectivity(method='distance', include_self=False, layout='dense')
            if cfg.dataset.connectivity.threshold:
                adj = adj * (dataset.dist < cfg.dataset.connectivity.threshold) 
            if cfg.dataset.connectivity.layout == 'edge_index':
                from tsl.ops.connectivity import adj_to_edge_index
                # adj_dense = adj
                adj = adj_to_edge_index(adj)
        else:
            adj = None

        # instantiate dataset
        # type ImputationDataset(SpatioTemporalDataset)
        # ImputationDataset.mask       --->  True = Valid and unmasked // False = Invalid or masked      (training_mask) == (mask & ~eval_mask)
        # ImputationDataset.eval_mask  --->  True = Valid and masked   // False = Invalid or unmasked    (eval_mask) == (mask & sampled_faults & all_VS)
        # ImputationDataset.val_mask   --->  True = Valid and val      // False = Invalid or train or test (mask & sampled_faults & val_VS)
        # ImputationDataset.test_mask  --->  True = Valid and test     // False = Invalid or train or val  (mask & sampled_faults & test_VS)
        torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          eval_mask=dataset.eval_mask,
                                          val_mask=dataset.val_mask,
                                          test_mask=dataset.test_mask,
                                          # covariates=covariates,
                                          transform=MaskInput(),
                                          connectivity=adj,
                                          window=cfg.window,
                                          stride=cfg.stride,
                                          window_lag=cfg.window_lag)

        # norm every channel independently across time and nodes
        scalers = {'target': StandardScaler(axis=(0, 1))}  # t, n

        # init the datamodule, no splitting along the temporal axis
        dm = SpatioTemporalDataModule(dataset=torch_dataset,
                                      scalers=scalers,
                                      splitter=CustomSplitter(val_split_fn=nodewise_index,
                                                              test_split_fn=nodewise_index,
                                                              mask_test_indices_in_val=False),
                                      batch_size=cfg.batch_size,
                                      workers=cfg.workers)
        # store name of nodes
        dm.nodes = dataset.nodes
        # store name of channels
        dm.channels = dataset.channels
        # store virtual sensors indices, for val and test
        dm.train_s = dataset.train_s
        dm.val_vs = dataset.val_vs
        dm.test_vs = dataset.test_vs

        # setup and fit scaler
        dm.setup()

        # if datamodule folder does not exist, create it
        if not 'datamodules' in os.listdir('.'): os.mkdir('datamodules')
        # datamodule will be saved with the following notation: {dataset_name}-{p_fault}-{p_noise}-va{val_size}-te{test_size}-s{seed}
        if not custom and not 'photovoltaic' in cfg.dataset.name:
            logger.info(f'DATAMODULE: saving datamodule to "datamodules/{cfg.datamodule_name}.pth"')
            torch.save(dm, f'datamodules/{cfg.datamodule_name}.pth')
        logger.info('DATAMODULE: done')

    else:
        logger.info(f'DATAMODULE: loading datamodule from "datamodules/{cfg.datamodule_name}.pth"')
        dm = torch.load(f'datamodules/{cfg.datamodule_name}.pth')

    # ########################################
    # # imputer                              #
    # ########################################
    imputer = init_module(dm, cfg)

    ########################################
    # logging options                      #
    ########################################
    if 'wandb' in cfg:
        exp_logger = WandbLogger(name=cfg.wandb.name+'-'+str(cfg.seed),
                                    save_dir=cfg.run.dir,
                                    offline=cfg.wandb.offline,
                                    project=cfg.wandb.project,
                                    group=cfg.wandb.name if cfg.group is None else cfg.group)
    elif 'tensorboard' in cfg:
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name='tensorboard')
    else:
        exp_logger = False
    if cfg.get('do_predict'): 
        if not exp_logger: raise AssertionError('Predictions are only allowed if logging is enabled')

    ########################################
    # training                             #
    ########################################
    trainer, checkpoint_callback = init_trainer(cfg, exp_logger)
    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################
    imputer.load_model(checkpoint_callback.best_model_path)
    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    # ########################################
    # # predict                              #
    # ########################################
    # this flag is usually False, and predictions are done in the 'run_eval.py' script
    if cfg.do_predict:
        output = trainer.predict(imputer, dataloaders=dm.test_dataloader(shuffle=False))
        logger.info('predicted')
        output = imputer.collate_prediction_outputs(output)
        output = torch_to_numpy(output)
        logger.info('collated')
        df_hat = {}
        df_hat['lower'], _ = aggregate_one_prediction(output['y_lower'], dm)
        df_hat['med'], _   = aggregate_one_prediction(output['y_hat'], dm)
        df_hat['upper'], _ = aggregate_one_prediction(output['y_upper'], dm)
        df_hat['true'], _  = aggregate_one_prediction(output['y'], dm)
        logger.info('aggregated')

        with open(f'{cfg.save_dir}/df_hat.pkl', 'wb') as f:
            pickle.dump({'df_hat':df_hat}, f)
            logger.info(f'predictions saved in {cfg.save_dir}/df_hat.pkl')
        assert dm.dataframe().equals(df_hat['true'])

    # close logger
    torch.cuda.empty_cache()
    if exp_logger: wandb.finish()

    logger.info('Script finished successfully')
    return True


def init_module(dm, cfg):
    # get model from name
    model_cls = get_model_class(cfg.model.name)

    # set the pinball flag
    if 'pinball' in cfg.model: 
        pinball = cfg.model.pinball
    else: 
        pinball = False

    # get settings and hparams
    model_kwargs = dict(n_nodes=dm.torch_dataset.n_nodes,
                        input_size=dm.torch_dataset.n_channels,
                        pinball=pinball)
    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    ########################################
    # loss function                        #
    ########################################
    if pinball:
        loss_fn = MaskedPinballLoss(q=None) # q is not set yet
    else:
        loss_fn = torch_metrics.MaskedMAE()

    ########################################
    # set metrics                          #
    ########################################
    # compute stds for the VRE metric
    df_torch = torch.from_numpy(dm.torch_dataset.numpy())       
    stds = torch.from_numpy(np.nanstd(df_torch, axis=0)).to('cuda' if torch.cuda.is_available() else 'cpu')
    if (stds==0).any(): raise ValueError('Some channel stds are zero, \
                                         this will cause division by zero in the VRE metric.\
                                         Please mask such channels or use a different metric.')    
    nchannels = dm.dataframe().columns.levels[1].nunique()
    log_metrics = {'mre': MultivariateMaskedMRE(dim=nchannels),
                   'vre': MaskedVRE(stds=stds),
                   'mae': torch_metrics.MaskedMAE()}
    if pinball:
        log_metrics['pb_low'] = MaskedPinballLoss(q=cfg.model.pinball_qs[0])
        log_metrics['pb_up'] = MaskedPinballLoss(q=cfg.model.pinball_qs[2])


    ########################################
    # Imputer                              #
    ########################################
    # setup imputer
    whiten_weight = (cfg.model.whiten_weight if 'whiten_weight' in cfg.model else 3.) # * cfg.model.whiten_prob_vs
    original_weight = 1 #1 - cfg.model.whiten_prob_vs
    imputer = Imputer(model_class=model_cls,
                      model_kwargs=model_kwargs,
                      optim_class=getattr(torch.optim, cfg.optimizer.name),
                      optim_kwargs=dict(cfg.optimizer.hparams),
                      loss_fn=loss_fn,
                      scale_target=cfg.scale_target,
                      metrics=log_metrics,
                      scheduler_class=scheduler_class,
                      scheduler_kwargs=scheduler_kwargs,

                      whiten_prob=cfg.model.whiten_prob,
                      whiten_prob_vs=cfg.model.whiten_prob_vs,
                      original_weight=original_weight,
                      whiten_weight=whiten_weight,

                      prediction_loss_weight=cfg.prediction_loss_weight,
                      impute_only_missing=cfg.impute_only_missing,
                      warm_up_steps=cfg.warm_up_steps,
                      pinball=pinball,
                      pinball_qs=cfg.model.pinball_qs if pinball else None,
                      reg_emb=cfg.model.reg_emb if 'reg_emb' in cfg.model else False)
    return imputer


def init_trainer(cfg, logger=None):
    # early stopping callback init
    early_stop_callback = EarlyStopping(monitor='val_mre',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(dirpath=cfg.run.dir,
                                          save_top_k=1,
                                          monitor='val_mre',
                                          mode='min')
    ########################################
    # Trainer                              #
    ########################################
    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.limit_train_batches,
                      limit_val_batches=cfg.limit_val_batches,
                      limit_test_batches=cfg.limit_test_batches,
                      default_root_dir=cfg.run.dir,
                      logger=logger,
                      #log_every_n_steps=len(dm.train_dataloader()),
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    return trainer, checkpoint_callback


if __name__ == '__main__':
    exp = Experiment(run_fn=run_train, config_path='config', config_name='config')
    res = exp.run()
