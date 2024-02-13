import torch
from torch import Tensor, nn
from typing import List, Optional, Tuple

from einops import rearrange

#import rnn
from tsl.nn.blocks.encoders.recurrent.base import RNNIBase
from tsl.nn.layers.recurrent import GRUCell, LSTMCell, StateType



class RNNI_withEmb(RNNIBase):
    """RNN encoder for sequences with missing data
    add the option to concat embeddings to the input before state update

    Args:
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = 0,
                 embedding_size: int = 0,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 concat_embeddings: bool = True,
                 unitary_mask: bool = False,
                 flip_time: bool = False,
                 n_layers: int = 1,
                 detach_input: bool = False,
                 cat_states_layers: bool = False,
                 dropout: float = 0.0):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mask_size = 1 if unitary_mask else input_size
        self.embedding_size = embedding_size

        # handle sizes
        if concat_embeddings:
            input_size = input_size + embedding_size
        if concat_mask:
            input_size = input_size + self.mask_size
        input_size = input_size + exog_size

        # init cells
        if cell == 'gru':
            cell = GRUCell
        elif cell == 'lstm':
            cell = LSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')
        cells = [
            cell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(n_layers)]

        super(RNNI_withEmb, self).__init__(cells, detach_input, concat_mask, flip_time,
                                           cat_states_layers)
        
        self.detach_input = detach_input
        self.concat_mask = concat_mask
        self.concat_embeddings = concat_embeddings
        self.flip_time = flip_time

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_size, self.input_size)

    def state_readout(self, h: List[StateType]):
        return self.readout(h[-1])
    
    def preprocess_input(self,
                        x: Tensor,
                        x_hat: Tensor,
                        input_mask: Tensor,
                        embeddings: Tensor,  # must be shape ((b n) f)
                        step: int,
                        *args,
                        u: Optional[Tensor] = None,
                        h: Optional[List[StateType]] = None,
                        **kwargs):
        if self.detach_input:
            x_hat = x_hat.detach()
        # put predictions in true values, where mask is False  
        x_t = torch.where(input_mask, x, x_hat)       
        # concat mask
        if self.concat_mask:
            x_t = torch.cat([x_t, input_mask], -1) 
        # concat embeddings
        if self.concat_embeddings:
            x_t = torch.cat([x_t, embeddings], -1)
        # concat exogenous variables
        if u is not None:
            x_t = torch.cat([x_t, u[:, step]], -1)
        return x_t
    
    def single_pass(self, x: Tensor, h: List[StateType], *args,
                **kwargs) -> List[StateType]:
        return super().single_pass(x, h)
    
    def forward(self,
                x: Tensor,
                input_mask: Tensor,
                embeddings: Tensor,
                *args,
                u: Optional[Tensor] = None,
                h: Optional[List[StateType]] = None,
                **kwargs) -> Tuple[Tensor, Tensor, List[StateType]]:
        """"""
        # x: [batch, time, *, features]
        if h is None:
            h = self.initialize_state(x)
        elif not isinstance(h, list):
            h = [h]
        # temporal conv
        h_out, x_out = [], []
        steps = x.size(1)
        steps = range(steps) if not self.flip_time else range(
            steps - 1, -1, -1)
        for step in steps:
            # readout phase
            x_hat = self.state_readout(h)
            x_out.append(x_hat)
            h_out.append(super().postprocess_state(h))
            # preprocess input to rnn's cell given readout
            x_t = self.preprocess_input(x[:, step], x_hat, 
                                        input_mask[:, step], embeddings,
                                        step, *args, **kwargs)
            h = self.single_pass(x_t, h, *args, **kwargs)

        if self.flip_time:
            x_out, h_out = x_out[::-1], h_out[::-1]

        x_out = torch.stack(x_out, dim=1)  # [(b n) t f]
        h_out = self.dropout(torch.stack(h_out, dim=1))  # [(b n) t f]
        h_last = self.dropout(super().postprocess_state(h))

        return x_out, h_out, h_last