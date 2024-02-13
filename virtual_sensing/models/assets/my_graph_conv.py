from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch.nn import Parameter
from torch_geometric.typing import OptTensor
from einops import rearrange, repeat, einsum

from tsl.nn.layers.multi import MultiLinear

from tsl.nn.utils import get_functional_activation

class myGraphConv(nn.Module):
    r""" The Graph Convolution layer with dense adjacency matrix, 
    implemented with matrix multiplication. """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation=None,
                 root_weight=True,
                 root_weight_type='linear',
                 bias=True,
                 mode: str = 'single', # or "multi"
                 n_instances: int = 1,  
                 pattern: str = None,
                 instance_dim: int = -2,
                 channel_dim: int = -1
                 ):
        super(myGraphConv, self).__init__()
        self.mode = mode
        self.root_weight = root_weight

        self.activation = get_functional_activation(activation)

        # Gconv mode, single or multi (separate weights for each channel)
        if mode == 'single':
            self.lin = nn.Linear(input_size, output_size, bias=False)
        elif mode == 'multi':
            self.lin = MultiLinear(input_size, output_size, n_instances,
                                   pattern=pattern,
                                   instance_dim=instance_dim,
                                   channel_dim=channel_dim,
                                   bias=False)
        else:
            raise NotImplementedError(f'mode: {mode} not implemented.')

        # Residual connection, identity or linear (single or multi)
        if root_weight:
            if root_weight_type == 'linear':
                if mode == 'single':
                    self.residual = nn.Linear(input_size, output_size, bias=False)
                elif mode == 'multi':
                    self.residual = MultiLinear(input_size, output_size, n_instances,
                                                pattern=pattern,
                                                instance_dim=instance_dim,
                                                channel_dim=channel_dim,
                                                bias=False)
            elif root_weight_type == 'identity':
                self.residual = nn.Identity()
            else:
                raise NotImplementedError(f'residual: {root_weight_type} not implemented.')
        else:
            self.register_parameter('residual', None)

        # Bias, single or multi
        if bias:
            if mode == 'single':
                self.bias = Parameter(torch.Tensor(output_size))
            elif mode == 'multi':
                self.bias = Parameter(torch.Tensor(n_instances, output_size))
        else:
            self.register_parameter('bias', None)  

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.residual is not None:
            self.residual.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self,
                x: Tensor,
                adj: Tensor,
                ) -> Tensor:
        """"""
        if self.mode == 'single':
            if len(x.shape) == 3:        
                temp = einsum(adj, self.lin(x), 'n m, b n h -> b m h')    # Ã ∙ (X ∙ 𝛩) 
            elif len(x.shape) == 4:
                temp = einsum(adj, self.lin(x), 'n m, b t n h -> b t m h')
            else:
                raise NotImplementedError(f'x: {x.shape} not implemented.')
            
        elif self.mode == 'multi':
            if len(x.shape) == 5:
                temp = einsum(adj, self.lin(x), 'n m, b t n f h -> b t m f h')   
            else:
                raise NotImplementedError(f'x: {x.shape} not implemented.')
            
        else:
            raise NotImplementedError(f'mode: {self.mode} not implemented.')

        if self.residual is not None:
            temp += self.residual(x)

        if self.bias is not None:
            temp += self.bias

        return self.activation(temp)                # activation    