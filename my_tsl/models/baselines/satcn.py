# tsl imports
from typing import Optional, Union
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

from tsl.nn.models.base_model import BaseModel

# SATCN imports
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from SATCN.model import tcn_layer
from SATCN.model import STower




class SATCN(BaseModel):

    """
    """

    def __init__(self, 
                 input_size: int,
                 avg_d: Tensor,   #average of the ``log(Degree + 1)`` in the training set
                 hidden_size: int = 64,
                 layers = 1, 
                 t_kernel = 2, 
                 aggragators = ['mean', 'softmin', 'softmax', 'normalised_mean', 'std'], 
                 scalers = ['identity', 'amplification', 'attenuation'], 
                 masking = True, 
                 dropout = 0,
                 pinball: bool = True):
        super(SATCN, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        avg_d = {'log':avg_d}

        self.s_layer0 = STower(input_size, 
                               hidden_size, 
                               aggragators + ['distance','d_std'], 
                               scalers,
                               avg_d, 
                               device, 
                               masking, 
                               dropout)
        self.t_layer0 = tcn_layer(t_kernel, 
                                  hidden_size, 
                                  hidden_size, 
                                  dropout)
        self.s_convs = nn.ModuleList()
        self.t_convs = nn.ModuleList()
        self.layers = layers
        for i in range(layers):
            self.s_convs.append(STower(hidden_size, hidden_size, aggragators, scalers, avg_d, device, False, dropout))
            self.t_convs.append(tcn_layer(t_kernel, hidden_size, hidden_size, dropout))

        out_size = 3*input_size if pinball else input_size
        self.out_conv = nn.Conv2d(hidden_size, out_size, (1, 1), 1)



    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                mask: OptTensor = None,
                u: OptTensor = None) -> list:
        """"""
        # x: [batch, time, nodes, features]
        raise NotImplementedError
        x = self.s_layer0(x, edge_index)
        x = self.t_layer0(x)
        for i in range(self.layers):
            x = self.s_convs[i](x, edge_index)
            x = self.t_convs[i](x)
        y = self.out_conv(x)
        return y

        return [[x_hat], []]

    def predict(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                mask: OptTensor = None,
                u: OptTensor = None) -> Tensor:
        """"""
        return self.forward(x = x, 
                            edge_index = edge_index,
                            edge_weight = edge_weight,
                            mask = mask, 
                            u = u)