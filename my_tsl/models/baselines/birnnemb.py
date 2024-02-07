from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.typing import OptTensor
from einops import rearrange, repeat

from tsl.nn.blocks import MLP
from tsl.nn.models.base_model import BaseModel

# embeddings
from tsl.nn.layers.base.embedding import NodeEmbedding

# time
from my_tsl.models.assets.rnn_encoders_embeddings import RNNI_withEmb

class BiRNNImputerEmbModel(BaseModel):

    """
    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        exog_size (int): Size of the optional exogenous variables.
            (default: ``0.``)
        cell (str): Type of recurrent cell to be used, one of [:obj:`gru`,
            :obj:`lstm`].
            (default: :obj:`gru`)
        concat_mask (bool): If :obj:`True`, then the input tensor is
            concatenated to the mask when fed to the RNN cell.
            (default: :obj:`True`)
        unitary_mask (bool): If :obj:`True`, then the mask is a single value
            and applies to all features.
            (default: :obj:`False`)
        flip_time (bool): If :obj:`True`, then the time is folded in the
            backward direction.
            (default: :obj:`False`)
        n_layers (int, optional): Number of hidden layers.
            (default: :obj:`1`)
        detach_input (bool): If :obj:`True`, call :meth:`~torch.Tensor.detach`
            on predictions before they are used to fill the gaps, breaking the
            error backpropagation.
            (default: :obj:`False`)
        cat_states_layers (bool): If :obj:`True`, then the states of the RNN are
            concatenated together.
            (default: :obj:`False`)
    """

    def __init__(self,
                 input_size: int,
                 rnn_hidden_size: int = 64,
                 embedding_size: int = 64,
                 exog_size: int = 0,

                 n_rnn_layers: int = 1,
                 cat_emb_rnn: bool = True,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 detach_input: bool = False,
                 cat_states_layers: bool = False,

                 n_nodes: Optional[int] = None,
                 
                 readout_mode: str = 'mlp',
                 n_mlp_layers: int = 1,
                 mlp_hidden_size: int = 128,
                 mlp_activation = 'relu',
                 cat_emb_out: bool = True,

                 pinball: bool = False):
        super(BiRNNImputerEmbModel, self).__init__()

        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.exog_size = exog_size

        self.n_rnn_layers = n_rnn_layers
        self.cat_emb_rnn = cat_emb_rnn
        self.concat_mask = concat_mask
        self.detach_input = detach_input

        self.readout_mode = readout_mode
        self.cat_emb_out = cat_emb_out

        self.pinball = pinball

        #### embeddings #### 
        self.emb = NodeEmbedding(n_nodes, embedding_size)

        #### time component #### 
        self.fwd_rnn = RNNI_withEmb(input_size=input_size,
                                    hidden_size=rnn_hidden_size,
                                    exog_size=exog_size,
                                    embedding_size=embedding_size,
                                    cell=cell,
                                    concat_mask=concat_mask,
                                    concat_embeddings=cat_emb_rnn,
                                    n_layers=n_rnn_layers,
                                    detach_input=detach_input,
                                    cat_states_layers=cat_states_layers,
                                    dropout=0.0)  
        self.bwd_rnn = RNNI_withEmb(input_size=input_size,
                                    hidden_size=rnn_hidden_size,
                                    exog_size=exog_size,
                                    embedding_size=embedding_size,
                                    cell=cell,
                                    concat_mask=concat_mask,
                                    concat_embeddings=cat_emb_rnn,
                                    flip_time=True,
                                    n_layers=n_rnn_layers,
                                    detach_input=detach_input,
                                    cat_states_layers=cat_states_layers,
                                    dropout=0.0)
        hidden_size = 2 * rnn_hidden_size * (n_rnn_layers if cat_states_layers else 1)

        if readout_mode == 'linear':
            self.readout = nn.Linear(hidden_size + embedding_size if cat_emb_out else hidden_size, 
                                     3 * input_size if pinball else input_size)
        elif readout_mode == 'mlp':
            #### MLP decoder #### 
            # one single decoder
            self.readout = MLP(input_size = hidden_size + embedding_size if cat_emb_out else hidden_size,
                               hidden_size = mlp_hidden_size,
                               output_size = 3 * input_size if pinball else input_size,
                               activation = mlp_activation,
                               n_layers = n_mlp_layers)
        else:
            raise ValueError(f'Unknown readout mode: {readout_mode}')


    def forward(self,
                x: Tensor,
                mask: OptTensor = None,
                sampled_idx = [],
                u: Optional[Tensor] = None) -> Union[Tensor, list]:
        """"""
        # x: [batch, time, nodes, features]
        batches, steps, nodes, features = x.size()
        # sample embeddings from node indices
        if sampled_idx: embs = self.emb()[sampled_idx, :]
        else:           embs = self.emb()
        
        # prepare inputs for RNN
        x = rearrange(x, f'b t n f -> (b n) t f')
        mask = rearrange(mask, f'b t n f -> (b n) t f')
        embs_rnn = repeat(embs, f'n f -> (b n) f', b = batches)

        if u is not None:
            if u.ndim == 3:  # no fc and 'b t f'
                u = repeat(u, f'b t f -> (b n) t f', n = nodes)
            else:  # no fc and 'b t n f'
                u = rearrange(u, f'b t n f -> (b n) t f')

        ########################################
        # Time component                       #
        ########################################
        _, h_fwd, _ = self.fwd_rnn(x, mask, embs_rnn, u)
        _, h_bwd, _ = self.bwd_rnn(x, mask, embs_rnn, u)

        h_rnn = torch.cat([h_fwd, h_bwd], axis = -1)

        ########################################
        # Readout                              #
        ########################################  
        if self.cat_emb_out:
            h_rnn = torch.cat([h_rnn, repeat(embs_rnn, f'b f -> b t f', t = steps)], axis = -1)
        x_hat = self.readout(h_rnn)

        x_hat = rearrange(x_hat, f'(b n) t f -> b t n f', b = batches)
        if self.pinball:
            x_hat = list(torch.split(x_hat, features, dim = -1))
        else:
            x_hat = [x_hat]

        return [x_hat, []]

    def predict(self,
                x: Tensor,
                mask: OptTensor = None,
                sampled_idx = [],
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x = x, 
                            mask = mask,
                            sampled_idx = sampled_idx,
                            u = u)
