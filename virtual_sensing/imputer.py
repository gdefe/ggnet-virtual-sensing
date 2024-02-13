from typing import Callable, List, Mapping, Optional, Tuple, Type, Union
from functools import partial

import torch
from torch import Tensor
from torch_geometric.data.storage import recursive_apply
from torchmetrics import Metric

from tsl.engines.predictor import Predictor

from virtual_sensing.node_sampler import sample_nodes #,sample_nodes_groups, sample_nodes_rnd
from utils import baselines_loss, format_output, unpack_output  

class Imputer(Predictor):
    r""":class:`~pytorch_lightning.core.LightningModule` to implement imputers.

    An imputer is an engines designed to fill out missing values in
    spatiotemporal data.

    Args:
        model (torch.nn.Module, optional): Model implementing the imputer.
            Ignored if argument `model_class` is not null. This argument should
            mainly be used for inference.
            (default: :obj:`None`)
        model_class (type, optional): Class of :obj:`~torch.nn.Module`
            implementing the imputer. If not `None`, argument `model` will be
            ignored.
            (default: :obj:`None`)
        model_kwargs (mapping, optional): Dictionary of arguments to be
            forwarded to :obj:`model_class` at instantiation.
            (default: :obj:`None`)
        optim_class (type, optional): Class of :obj:`~torch.optim.Optimizer`
            implementing the optimizer to be used for training the model.
            (default: :obj:`None`)
        optim_kwargs (mapping, optional): Dictionary of arguments to be
            forwarded to :obj:`optim_class` at instantiation.
            (default: :obj:`None`)
        loss_fn (callable, optional): Loss function to be used for training the
            model.
            (default: :obj:`None`)
        scale_target (bool): Whether to scale target before evaluating the loss.
            The metrics instead will always be evaluated in the original range.
            (default: :obj:`False`)
        whiten_prob (float or list): Randomly mask out a valid datapoint during
            a training step with probability :obj:`whiten_prob`. If a list is
            passed, :obj:`whiten_prob` is sampled from the list for each batch.
            (default: :obj:`0.05`)
        prediction_loss_weight (float): The weight to assign to predictions
            (if any) in the loss. The loss is computed as

            .. math::

                L = \ell(\bar{y}, y, m) + \lambda \sum_i \ell(\hat{y}_i, y, m)

            where :math:`\ell(\bar{y}, y, m)` is the imputation loss,
            :math:`\ell(\bar{y}_i, y, m)` is the forecasting error of prediction
            :math:`\bar{y}_i`, and :math:`\lambda` is
            :obj:`prediction_loss_weight`.
            (default: :obj:`1.0`)
        impute_only_missing (bool): Whether to impute only missing values in
            inference or the whole sequence.
            (default: :obj:`True`)
        warm_up_steps (int, tuple): Number of steps to be considered as warm up
            stage at the beginning of the sequence. If a tuple is provided, the
            padding is applied both at the beginning and the end of the
            sequence.
            (default: :obj:`0`)
        metrics (mapping, optional): Set of metrics to be logged during
            train, val and test steps. The metric's name will be automatically
            prefixed with the loop in which the metric is computed (e.g., metric
            :obj:`mae` will be logged as :obj:`train_mae` when evaluated during
            training).
            (default: :obj:`None`)
        scheduler_class (type): Class of
            :obj:`~torch.optim.lr_scheduler._LRScheduler` implementing the
            learning rate scheduler to be used during training.
            (default: :obj:`None`)
        scheduler_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`scheduler_class` at instantiation.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        loss_fn: Optional[Callable] = None,
        scale_target: bool = False,
        metrics: Optional[Mapping[str, Metric]] = None,
        *,
        whiten_prob: Optional[Union[float, List[float]]] = 0.05,
        whiten_prob_vs: Optional[Union[float, List[float]]] = 0.05,
        original_weight: float = 1.0,
        whiten_weight: float = 1.0,
        prediction_loss_weight: float = 1.0,
        impute_only_missing: bool = True,
        warm_up_steps: Union[int, Tuple[int, int]] = 0,
        model_class: Optional[Type] = None,
        model_kwargs: Optional[Mapping] = None,
        optim_class: Optional[Type] = None,
        optim_kwargs: Optional[Mapping] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
        pinball: bool = False,
        pinball_qs: Optional[List[float]] = None,
        reg_emb: bool = False,
    ):
        super(Imputer, self).__init__(model=model,
                                      model_class=model_class,
                                      model_kwargs=model_kwargs,
                                      optim_class=optim_class,
                                      optim_kwargs=optim_kwargs,
                                      loss_fn=loss_fn,
                                      scale_target=scale_target,
                                      metrics=metrics,
                                      scheduler_class=scheduler_class,
                                      scheduler_kwargs=scheduler_kwargs)

        if isinstance(whiten_prob, (list, tuple)):
            self.whiten_prob = torch.tensor(whiten_prob)
        else:
            self.whiten_prob = whiten_prob

        if isinstance(whiten_prob_vs, (list, tuple)):
            self.whiten_prob_vs = torch.tensor(whiten_prob_vs)
        else:
            self.whiten_prob_vs = whiten_prob_vs
        self.original_weight = original_weight
        self.whiten_weight = whiten_weight

        self.prediction_loss_weight = prediction_loss_weight
        self.impute_only_missing = impute_only_missing
        self.pinball = pinball
        self.pinball_qs = pinball_qs
        self.reg_emb = reg_emb
        
        if isinstance(warm_up_steps, int):
            self.warm_up_steps = (warm_up_steps, 0)
        elif isinstance(warm_up_steps, (list, tuple)):
            self.warm_up_steps = tuple(warm_up_steps)
        if len(self.warm_up_steps) != 2:
            raise ValueError(
                "'warm_up_steps' must be an int of time steps to "
                "be cut at the beginning of the sequence or a "
                "pair of int if the sequence must be trimmed in a "
                "bidirectional way.")

    def trim_warm_up(self, *args):
        """Trim all tensors in :obj:`args` removing a number of first and last
        steps equals to :obj:`(self.warm_up_steps[0], self.warm_up_steps[1])`,
        respectively."""
        left, right = self.warm_up_steps
        # assume time in second dimension (after batch dim)
        trim = lambda s: s[:, left:s.size(1) - right]  # noqa
        args = recursive_apply(args, trim)
        if len(args) == 1:
            return args[0]
        return args
    

    def get_loss(self, y_hat_loss, y_loss, mask):
        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        # baselines
        if self.model.__class__.__name__ == 'BRITSModel' or \
           self.model.__class__.__name__ == 'SAITSModel':
            loss  = baselines_loss(y_hat_loss, y_loss, mask)['total_loss']

        # our methods
        else:
            # loss computation
            loss = torch.tensor(0., device=self.device)
            imputations, predictions = y_hat_loss
            if len(imputations) == 1: # imputation and predictions
                loss += self.loss_fn(imputations[0], y_loss, mask)
                for pred in predictions:
                    pred_loss = self.loss_fn(pred, y_loss, mask)
                    loss += self.prediction_loss_weight * pred_loss

            elif len(imputations) == 3 and self.pinball: # 3 pinball components
                for i, imp in enumerate(imputations):
                    self.update_metric_fn_kwargs(q = self.pinball_qs[i])
                    loss += self.loss_fn(imp, y_loss, mask)
                self.update_metric_fn_kwargs(q = 0.5)
                for pred in predictions:
                    pred_loss = self.loss_fn(pred, y_loss, mask)
                    loss += self.prediction_loss_weight * pred_loss
            else:    
                raise ValueError('get_loss: Something is wrong with the number of returned values \
                                  from the model or the pinball settings.')
            # regularization
            if self.reg_emb:
                if hasattr(self.model, 'emb_G'):
                    sep_term, term = self.model.emb_G.clustering_loss()
                    loss += term
                elif hasattr(self.model, 'emb'):
                    sep_term, term = self.model.emb.clustering_loss()
                    loss += term

        return loss
    
    def update_metrics(self, metrics, lower, median, upper, 
                       y, mask, sampled_idx):
        to_update = [m.replace(metrics.prefix, '') for m in metrics.keys() if 'pb' not in m]
        # if node sampling, keep stds only of sampled nodes (Not implemented yet)
        # if 'vre' in to_update and self.sampler is not None: 
        #     metrics['vre'].stds_idx = metrics['vre'].stds[:, :, sampled_idx, :]
        for m in to_update:
            metrics[m].update(median, y, mask)  
        if self.pinball: 
            metrics['pb_low'].update(lower, y, mask) 
            metrics['pb_up'].update(upper, y, mask)


    # Imputation data hooks ###################################################
    def update_metric_fn_kwargs(self, **metric_fn_kwargs):
        self.loss_fn.metric_fn = partial(self.loss_fn.metric_fn, **metric_fn_kwargs)
        
    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        r"""For every training batch, randomly mask out sensors with probability
        :obj:`p = self.whiten_prob`. Then, whiten missing values in
        :obj:`batch.input.x`."""
        super(Imputer, self).on_train_batch_start(batch, batch_idx)
        if self.whiten_prob is not None:
            # randomly mask out value with probability p = whiten_prob
            mask = batch.mask
            batch.original_mask = mask
            p = self.whiten_prob
            p_vs = self.whiten_prob_vs
            if isinstance(p, Tensor) and p.ndim > 0:
                # broadcast p to mask size
                p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
                # sample p for each batch
                p = p[torch.randint(len(p), p_size)].to(device=mask.device)
            # set each non-zero element of mask to 0 with probability p
            whiten_mask = torch.rand(mask.size(), device=mask.device) > p

            # sample full sensors whitening mask
            b,t,n,d = mask.size()
            # sample some virtual sensors among (b, n, d)
            vs = torch.rand([b,n,d], device=mask.device) > p_vs # (b, n, d), False if it will be masked
            # expand along time dimension
            whiten_mask_vs = vs.unsqueeze(1).expand(-1,t,-1,-1) # (b, t, n, d)

            batch.mask = mask & whiten_mask & whiten_mask_vs
        
            # whiten missing values
            if 'x' in batch.input:
                batch.input.x = batch.input.x * batch.mask

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Make predictions
        output = self.predict(**batch.input)
        y_hat, _ = format_output(output)
        y_hat, _ = y_hat

        # Rescale outputs
        if self.scale_target:
                y_hat = batch.transform['y'].inverse_transform(y_hat)

        # # fill missing values in target data
        # if self.impute_only_missing:
        #     y_hat[i] = torch.where(batch.mask.bool(), batch.y, y_hat[i])

        for i in range(len(y_hat)):
            y_hat[i] = y_hat[i].detach()

        # return dict
        output = dict(**batch.target,
                      y_lower = y_hat[0] if self.pinball else torch.zeros_like(y_hat[0]),
                      y_hat   = y_hat[1] if self.pinball else y_hat[0],
                      y_upper = y_hat[2] if self.pinball else torch.zeros_like(y_hat[0]),
                      mask=batch.mask,
                      eval_mask=batch.eval_mask,
                      val_mask=batch.val_mask,
                      test_mask=batch.test_mask)
        return output

    def shared_step(self, batch, mask):
        y = y_loss = batch.y
        output = self.predict_batch(batch, 
                                    preprocess=False, 
                                    postprocess=not self.scale_target)
        # from various output to [[imp1, imp2, imp3], [pred1, pred2, ...]]
        y_hat, y_hat_loss = format_output(output)  

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y_loss)
            y_hat = batch.transform['y'].inverse_transform(y_hat)
        
        # whiten mask weighting in loss calculation
        if hasattr(batch, 'original_mask'):
            original_loss = self.get_loss(y_hat_loss, y_loss, batch.mask)
            whiten_loss = self.get_loss(y_hat_loss, y_loss, batch.original_mask & ~batch.mask)
            loss = (original_loss * self.original_weight) + (whiten_loss * self.whiten_weight)
        else:
            loss = self.get_loss(y_hat_loss, y_loss, mask)
        y_hat = [[elem.detach() for elem in tup] for tup in y_hat]
        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        ### nodes sampling is not implemented yet, always take all nodes
        # if self.sampler=='random':
        #     sampled_idxs = [sample_nodes_rnd(batch, self.sample_n_nodes)]
        # elif self.sampler=='groups':
        #     sampled_idxs = sample_nodes_groups(batch, self.sample_n_nodes)
        # else:
        sampled_idxs = [range(batch.num_nodes)]

        # loop over sampled nodes
        loss_tot = torch.tensor(0., device=self.device)
        for sampled_idx in sampled_idxs:
            batch_s = sample_nodes(batch, sampled_idx)
            # forward pass
            y_hat, y, loss = self.shared_step(batch_s, batch_s.original_mask)
            y_hat, _ = y_hat # discard predictions, keep imputations
            lower, median, upper = unpack_output(y_hat, self.pinball)
            # Logging
            self.update_metrics(self.train_metrics, lower, median, upper, 
                                y, batch_s.original_mask, sampled_idx)
            loss_tot += loss

        loss_tot /= len(sampled_idxs)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss_tot, batch_size=batch.batch_size)
        return loss_tot

    def validation_step(self, batch, batch_idx):
        ### nodes sampling is not implemented yet, always take all nodes
        # if self.sampler=='random' or self.sampler=='groups':
        #     sampled_idxs = sample_nodes_groups(batch, self.sample_n_nodes)
        # else:
        sampled_idxs = [range(batch.num_nodes)]

        val_loss_tot = torch.tensor(0., device=self.device)
        for sampled_idx in sampled_idxs:
            batch_s = sample_nodes(batch, sampled_idx)
            # forward pass
            y_hat, y, val_loss = self.shared_step(batch_s, batch_s.val_mask)
            y_hat, _ = y_hat # discard predictions
            lower, median, upper = unpack_output(y_hat, self.pinball)
            # Logging
            self.update_metrics(self.val_metrics, lower, median, upper, 
                                y, batch_s.val_mask, sampled_idx)
            val_loss_tot += val_loss

        val_loss_tot /= len(sampled_idxs)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss_tot, batch_size=batch.batch_size)
        return val_loss_tot

    def test_step(self, batch, batch_idx):
        ### nodes sampling is not implemented yet, always take all nodes
        # if self.sampler=='random' or self.sampler=='groups':
        #     sampled_idxs = sample_nodes_groups(batch, self.sample_n_nodes)
        # else:
        sampled_idxs = [range(batch.num_nodes)]

        test_loss = torch.tensor(float('nan'), device=self.device)
        for sampled_idx in sampled_idxs:
            batch_s = sample_nodes(batch, sampled_idx)
            # output is in original scale
            output = self.predict_step(batch_s, batch_idx)
            # prepare for metrics computation
            y = batch_s.y
            if self.pinball:
                lower, median, upper = [output['y_lower'], output['y_hat'], output['y_upper']]
            else:
                lower, median, upper = None, output['y_hat'], None

            # Logging
            self.update_metrics(self.test_metrics, lower, median, upper, 
                                y, batch_s.test_mask, sampled_idx)
            
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss