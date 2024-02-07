from typing import Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.typing import OptTensor
from einops import rearrange, repeat, einsum

from tsl.nn.blocks import MLP
from tsl.nn.models.base_model import BaseModel

# time
from my_tsl.models.assets.rnn_encoders_embeddings import RNNI_withEmb

# space
from my_tsl.models.assets.my_graph_conv import myGraphConv
from tsl.nn.utils import get_functional_activation
from my_tsl.models.assets.embeddings import ClusterizedNodeEmbedding
# from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.layers.graph_convs import GraphConv, GatedGraphNetwork

class TimeThenSpaceModel(BaseModel):

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
                 mlp_hidden_size: int = 128,
                 embedding_size: int = 16,
                 exog_size: int = 0,

                 n_rnn_layers: int = 1,
                 n_conv_layers: int = 1,
                 n_mlp_layers: int = 1,
                 
                 rnn_bidir: bool = True,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 detach_input: bool = False,
                 cat_states_layers: bool = False,

                 gnn_mode = 'manual',   # manual / conv / gated
                 gnn_root_weight = True,
                 gnn_root_weight_type = 'linear',
                 gnn_activation = None,
                 gnn_dropout: float = 0.,
                 n_nodes: Optional[int] = None,
                 
                 embedding_h_size: int = 16,
                 cat_emb_rnn: bool = True,
                 cat_emb_gnn: bool = False,
                 cat_emb_out: bool = True,
                 weight_adj_gnn: bool = True,
                 n_clusters: int = 5,

                 merge_mode: str = 'cat',
                 mlp_activation: str = 'relu',

                 pinball: bool = False):
        super(TimeThenSpaceModel, self).__init__()

        self.gnn_mode = gnn_mode

        self.cat_emb_rnn = cat_emb_rnn
        self.cat_emb_gnn = cat_emb_gnn
        self.cat_emb_out = cat_emb_out
        self.weight_adj_gnn = weight_adj_gnn
        self.merge_mode = merge_mode

        self.pinball = pinball

        #### embeddings #### 
        # regularization (K = 5)
        self.emb = ClusterizedNodeEmbedding(n_nodes, embedding_size, 
                                              n_clusters=n_clusters, tau=1.0,
                                              separation_loss=False, sep_eps=1.)
        self.emb.init_centroids()
        # MLP for embeddings
        self.emb_mlp_1 = MLP(input_size = embedding_size, 
                             hidden_size = embedding_h_size, 
                             output_size = embedding_h_size,
                             activation = 'tanh',
                             n_layers=1)
        self.emb_mlp_2 = MLP(input_size = embedding_size, 
                             hidden_size = embedding_h_size, 
                             output_size = embedding_h_size,
                             activation = 'tanh',
                             n_layers=1)

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

        #### space component ####
        # (keep same dimesionality as RNN)
        if gnn_mode == 'manual':
            self.gnn_layers = nn.ModuleList([
                myGraphConv(hidden_size + embedding_size if (cat_emb_gnn and l == 0) else hidden_size,
                            hidden_size,
                            bias=True,
                            root_weight=gnn_root_weight,
                            root_weight_type=gnn_root_weight_type,
                            activation=gnn_activation)
                for l in range(n_conv_layers)])
        # elif gnn_mode == 'conv':
        #     self.gnn_layers = nn.ModuleList([
        #         GraphConv(hidden_size + embedding_size if (cat_emb_gnn and l == 0) else hidden_size,
        #                   hidden_size,
        #                   bias = True,
        #                   norm = 'mean',
        #                   activation = gnn_activation)
        #         for l in range(n_conv_layers)])
            
        # elif gnn_mode == 'gated':
        #     self.gnn_layers = nn.ModuleList([
        #         GatedGraphNetwork(hidden_size + embedding_size if (cat_emb_gnn and l == 0) else hidden_size,
        #                           hidden_size,
        #                           activation = gnn_activation,
        #                           parametrized_skip_conn = False)
        #         for l in range(n_conv_layers)])
        else:
            raise ValueError(f'Unknown GNN mode {gnn_mode}')
        self.dropout = nn.Dropout(gnn_dropout)

        #### MLP decoder #### 
        # one single decoder
        if   merge_mode=='cat': dec_in_size = 2 * hidden_size
        elif merge_mode=='sum': dec_in_size = hidden_size
        else: raise ValueError(f'Unknown merge mode {merge_mode}')
        self.readout = MLP(input_size = dec_in_size + embedding_size if cat_emb_out else dec_in_size,
                           hidden_size = mlp_hidden_size,
                           output_size = 3 * input_size if pinball else input_size,
                           activation = mlp_activation,
                           n_layers = n_mlp_layers)

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
        # output is already [b t n f], [b t n h], [b n h] 
        # to emulate rnni and birnni: use x_hat directly and use 'return_predictions=True'
        # if not, use hidden states
        _, h_fwd, _ = self.fwd_rnn(x, mask, embs_rnn, u)
        _, h_bwd, _ = self.bwd_rnn(x, mask, embs_rnn, u)

        h_rnn = torch.cat([h_fwd, h_bwd], axis = -1)
        h_rnn = rearrange(h_rnn, f'(b n) t f -> b t n f', b = batches)

        ########################################
        # Space component                      #
        ########################################         
        # last state representation   # [b t n h] -> [b n h]
        # h_gnn = h_rnn[:,-1]
        # gnn along every time stamp
        h_gnn = rearrange(h_rnn, f'b t n f -> (b t) n f')

        # # # adjacency
        if self.weight_adj_gnn:
            emb_encoded_1 = self.emb_mlp_1(embs)
            emb_encoded_2 = self.emb_mlp_2(embs)
            adj = emb_encoded_1 @ emb_encoded_2.T    # calculate the adjacency matrix
            adj = F.softmax(adj, dim=1)       # normalize with softmax
        else:
            adj = torch.ones((nodes, nodes)).to(h_gnn.device)  # dense
        
        # # # convolutions: dense spatial attention layer
        if self.cat_emb_gnn:
            h_gnn = torch.cat([h_gnn, repeat(embs, f'n f -> b n f', b = batches*steps)], axis = -1)
        edge_index, edge_weight = dense_to_sparse(adj)
        for conv in self.gnn_layers:
            if self.gnn_mode == 'manual':
                h_gnn = conv(h_gnn, adj)                                 # MyGraphConv layer
            if self.gnn_mode == 'conv':
                h_gnn = conv(h_gnn, edge_index, edge_weight)             # GraphConv layer
            if self.gnn_mode == 'gated':
                h_gnn = conv(h_gnn, edge_index)                          # GatedGraphConv layer
            h_gnn = self.dropout(h_gnn) # [b n f] 
        # h_gnn = repeat(h_gnn, 'b n f -> b t n f', t = steps)
        h_gnn = rearrange(h_gnn, f'(b t) n f -> b t n f', t = steps)


        ########################################
        # Readout                              #
        ########################################  
        if   self.merge_mode == 'cat': h_combo = torch.cat([h_rnn, h_gnn], dim = -1)
        elif self.merge_mode == 'sum': h_combo = h_rnn + h_gnn
        else: h_combo = h_rnn
            
        if self.cat_emb_out:
            h_combo = torch.cat([h_combo, repeat(embs, f'n f -> b t n f', b = batches, t = steps)], axis = -1)

        x_hat = self.readout(h_combo)
        
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
