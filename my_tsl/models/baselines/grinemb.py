from typing import Optional, Union
from einops import repeat

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.typing import Adj, OptTensor

from tsl.nn.blocks import MLP
from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.models.base_model import BaseModel

from my_tsl.models.assets.grin_cell_embeddings import myGRINCell



class GRINEmbModel(BaseModel):
    r"""The Graph Recurrent Imputation Network with DCRNN cells from the paper
    `"Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural
    Networks" <https://arxiv.org/abs/2108.00298>`_ (Cini et al., ICLR 2022).
    """

    return_type = list

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 embedding_size: int = 16,

                 # readout
                 merge_mode: str = 'mlp',
                 out_h_size: int = 64,
                 out_dropout: float = 0.,

                 cat_emb_rnn: bool = True,
                #  cat_emb_gnn: bool = False,
                 cat_emb_out: bool = True,

                 exog_size: int = 0,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None,
                 layer_norm: bool = False,
                 dropout: float = 0.,

                 pinball: bool = False):
        super(GRINEmbModel, self).__init__()
        self.fwd_gril = myGRINCell(input_size=input_size,
                                 hidden_size=hidden_size,
                                 exog_size=exog_size,
                                 embedding_size=embedding_size,
                                 n_layers=n_layers,
                                 dropout=dropout,
                                 cat_emb_rnn=cat_emb_rnn,
                                 n_nodes=None,
                                 layer_norm=layer_norm)
        self.bwd_gril = myGRINCell(input_size=input_size,
                                 hidden_size=hidden_size,
                                 exog_size=exog_size,
                                 embedding_size=embedding_size,
                                 n_layers=n_layers,
                                 dropout=dropout,
                                 cat_emb_rnn=cat_emb_rnn,
                                 n_nodes=None,
                                 layer_norm=layer_norm)

        self.emb = NodeEmbedding(n_nodes, embedding_size)

        self.cat_emb_out = cat_emb_out
        self.pinball = pinball

        self.merge_mode = merge_mode
        if merge_mode == 'mlp':
            out_in_size = 4 * hidden_size + input_size
            self.readout = nn.Sequential(nn.Linear(out_in_size + embedding_size if cat_emb_out else out_in_size, 
                                                   out_h_size),
                                         nn.ReLU(), nn.Dropout(out_dropout),
                                         nn.Linear(out_h_size, 
                                                   3 * input_size if self.pinball else input_size))
        elif merge_mode in ['mean', 'sum', 'min', 'max']:
            self.readout = getattr(torch, merge_mode)
        else:
            raise ValueError("Merge option %s not allowed." % merge_mode)



    def forward(self,
                x: Tensor,
                mask: OptTensor = None,
                u: OptTensor = None) -> list:
        """"""
        # x: [batch, time, nodes, features]
        batches, steps, nodes, features = x.size()

        # adjacency
        adj = self.emb() @ self.emb().T      # calculate the adjacency matrix
        adj.fill_diagonal_(-float('inf'))    # put the diagonal to -inf    # is this better than softmax then 0?
        adj = F.softmax(adj, dim=0)          # normalize with softmax   
        # adj -= torch.diag(torch.diag(adj))   # put the diagonal to zero
        edge_index, edge_weight = dense_to_sparse(adj)
        embs_rnn = repeat(self.emb(), f'n f -> b n f', b = batches)
        
        """-----------------"""
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_gril(x,
                                                       embeddings=embs_rnn,
                                                       adj=adj,
                                                       mask=mask,
                                                       u=u)
        # Backward
        rev_x = x.flip(1)
        rev_mask = mask.flip(1) if mask is not None else None
        rev_u = u.flip(1) if u is not None else None
        *bwd, _ = self.bwd_gril(rev_x,
                                embeddings=embs_rnn,
                                adj=adj,
                                mask=rev_mask,
                                u=rev_u)
        bwd_out, bwd_pred, bwd_repr = [res.flip(1) for res in bwd]

        if self.merge_mode == 'mlp':
            inputs = [fwd_repr, bwd_repr, mask]
            if self.cat_emb_out:
                inputs.append( self.emb(expand=(batches, steps, -1, -1)) )
            imputation = torch.cat(inputs, dim=-1)
            imputation = self.readout(imputation)
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=-1)
            imputation = self.readout(imputation, dim=-1)

        """-----------------"""
        if self.pinball:
            x_hat = list(torch.split(imputation, features, dim = -1))
        else:
            x_hat = [imputation]

        return x_hat + [fwd_out, bwd_out, fwd_pred, bwd_pred]


    def predict(self,
                x: Tensor,
                mask: OptTensor = None,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x = x, 
                            mask = mask, 
                            u = u)