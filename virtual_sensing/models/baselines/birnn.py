from typing import Optional, Union

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from tsl import logger
from tsl.nn.blocks import MLP
from tsl.nn.blocks.encoders.recurrent import RNNI
from tsl.nn.models.base_model import BaseModel




class BiRNNImputerModel(BaseModel):
    r"""
    Bi-directional RNN imputer model.
    Args:
        input_size (int): Number of features per node
        hidden_size (int): Number of features in the hidden state
        exog_size (int): Number of features in the exogenous input
        cell (str): Type of recurrent cell to use
        concat_mask (bool): Whether to concatenate the mask to the input
        detach_input (bool): Whether to detach the input from the graph
        n_layers (int): Number of recurrent layers
        cat_states_layers (bool): Whether to concatenate the states of all layers
        dropout (float): Dropout probability
        readout_mode (str): Readout mode to use. Either 'linear' or 'mlp'
        n_mlp_layers (int): Number of layers in the MLP decoder
        mlp_hidden_size (int): Number of features in the hidden layers of the MLP decoder
        mlp_activation (str): Activation function to use in the MLP decoder
    """

    return_type = list

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 exog_size: int = 0,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 detach_input: bool = False,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 dropout: float = 0.,
                 
                 readout_mode: str = 'linear',
                 n_mlp_layers: int = 1,
                 mlp_hidden_size: int = 128,
                 mlp_activation = 'relu'):
        super(BiRNNImputerModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        self.concat_mask = concat_mask
        self.detach_input = detach_input
        self.n_layers = n_layers
        self.cat_states_layers = cat_states_layers

        self._to_pattern = '(b n) t f'

        self.fwd_rnn = RNNI(input_size=input_size,
                            hidden_size=hidden_size,
                            exog_size=exog_size,
                            cell=cell,
                            concat_mask=concat_mask,
                            n_layers=n_layers,
                            detach_input=detach_input,
                            cat_states_layers=cat_states_layers)
        self.bwd_rnn = RNNI(input_size=input_size,
                            hidden_size=hidden_size,
                            exog_size=exog_size,
                            cell=cell,
                            concat_mask=concat_mask,
                            flip_time=True,
                            n_layers=n_layers,
                            detach_input=detach_input,
                            cat_states_layers=cat_states_layers)

        self.dropout = nn.Dropout(dropout)

        out_size = hidden_size * (n_layers if cat_states_layers else 1)

        if readout_mode == 'linear':
            self.readout = nn.Linear(2 * out_size, self.input_size)
        elif readout_mode == 'mlp':
            #### MLP decoder #### 
            # one single decoder
            self.readout = MLP(input_size = 2 * out_size,
                               hidden_size = mlp_hidden_size,
                               output_size = self.input_size,
                               activation = mlp_activation,
                               n_layers = n_mlp_layers)
            
    def forward(self,
                x: Tensor,
                mask: Tensor,
                u: Optional[Tensor] = None) -> Union[Tensor, list]:
        """"""
        # x: [batch, time, nodes, features]
        nodes = x.size(2)

        x = rearrange(x, f'b t n f -> {self._to_pattern}')
        mask = rearrange(mask, f'b t n f -> {self._to_pattern}')

        if u is not None:
            if u.ndim == 3:  # no fc and 'b t f'
                u = repeat(u, f'b t f -> {self._to_pattern}', n=nodes)
            else:  # no fc and 'b t n f'
                u = rearrange(u, f'b t n f -> {self._to_pattern}')

        _, h_fwd, _ = self.fwd_rnn(x, mask, u)
        _, h_bwd, _ = self.bwd_rnn(x, mask, u)

        h_fwd = rearrange(h_fwd, f'{self._to_pattern} -> b t n f', n=nodes)
        h_bwd = rearrange(h_bwd, f'{self._to_pattern} -> b t n f', n=nodes)

        h = self.dropout(torch.cat([h_fwd, h_bwd], -1))
        x_hat = self.readout(h)

        return [[x_hat], []]

    def predict(self,
                x: Tensor,
                mask: Tensor,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x=x,
                            mask=mask,
                            u=u)
