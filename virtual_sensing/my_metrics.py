from typing import Any

import torch
from torch.nn import functional as F
from torchmetrics.utilities.checks import _check_same_shape

from tsl.metrics.torch.functional import pinball_loss, multi_quantile_pinball_loss
from tsl.metrics.torch.metric_base import MaskedMetric        


class MultivariateMaskedMRE(MaskedMetric):
    """Mean Relative Error Metric for Multivariate Data, 
    takes the average over channels.

    Args:
        dim (int): dimension to take the average over (usually the number of channels)
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 dim: int,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 **kwargs: Any):
        super(MultivariateMaskedMRE, self).__init__(metric_fn=F.l1_loss,
                                                    mask_nans=mask_nans,
                                                    mask_inf=mask_inf,
                                                    metric_fn_kwargs={'reduction': 'none'},
                                                    at=at,
                                                    **kwargs)
        self.dim = dim
        # numerator
        self.add_state('num',
                       dist_reduce_fx='sum',
                       default=torch.zeros(dim, dtype=torch.float))
        # denominator
        self.add_state('den',
                       dist_reduce_fx='sum',
                       default=torch.zeros(dim, dtype=torch.float))
        
    # only difference: sum over all but channels    
    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.zeros_like(val))
        y_masked = torch.where(mask, y, torch.zeros_like(y))
        return val.sum(axis=[0,1,2]), y_masked.sum(axis=[0,1,2])

    # only difference: sum over all but channels    
    def _compute_std(self, y_hat, y):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(axis=[0,1,2]), y.sum(axis=[0,1,2])

    def compute(self):
        val = self.num / self.den
        assert val.shape[0] == self.dim
        # finally, mean over channels
        return val.nansum() / torch.count_nonzero(~torch.isnan(val))

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            num, den = self._compute_masked(y_hat, y, mask)
        else:
            num, den = self._compute_std(y_hat, y)
        self.num += num
        self.den += den    
    
    







class MaskedVRE(MaskedMetric):
    """Mean Absolute Percentage Error Metric.

    Args:
        stds (torch.Tensor): Standard deviations of the target variables.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, 
                 stds,
                 mask_nans=False, 
                 mask_inf=False,
                 at=None, 
                 **kwargs: Any):
        super(MaskedVRE,self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        **kwargs)  
        self.stds = stds

    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        # using std
        val = val / self.stds
        val = torch.where(mask, val, torch.zeros_like(val))
        return val.sum(), mask.sum()
    
    def _compute_std(self, y_hat, y):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        val = val / self.stds
        return val.sum(), val.numel()
    



class MaskedPinballLoss(MaskedMetric):
    """Quantile loss.

    Args:
        q (float): Target quantile.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        compute_on_step (bool, optional): Whether to compute the metric
            right-away or if accumulate the results. This should be :obj:`True`
            when using the metric to compute a loss function, :obj:`False` if
            the metric is used for logging the aggregate error across different
            mini-batches.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 q,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 **kwargs: Any):
        super(MaskedPinballLoss,
              self).__init__(metric_fn=pinball_loss,
                             mask_nans=mask_nans,
                             mask_inf=mask_inf,
                             metric_fn_kwargs={'q': q},
                             at=at,
                             **kwargs)
    


class MaskedMultiPinballLoss(MaskedMetric):
    """Quantile loss.

    Args:
        qs (float): Target quantiles.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        compute_on_step (bool, optional): Whether to compute the metric
            right-away or if accumulate the results. This should be :obj:`True`
            when using the metric to compute a loss function, :obj:`False` if
            the metric is used for logging the aggregate error across different
            mini-batches.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 qs,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 **kwargs: Any):
        super(MaskedMultiPinballLoss,
              self).__init__(metric_fn=multi_quantile_pinball_loss,
                             mask_nans=mask_nans,
                             mask_inf=mask_inf,
                             metric_fn_kwargs={'qs': qs},
                             at=at,
                             **kwargs)
    
    