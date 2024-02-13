from typing import Optional
from einops import repeat, rearrange

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.typing import OptTensor

from tsl.nn.models.base_model import BaseModel
from tsl.nn.layers.norm import LayerNorm

# temporal
from tsl.nn.layers.base import Dense
from tsl.nn.layers.multi.conv import MultiTemporalConv
# graph
from virtual_sensing.models.assets.embeddings import ClusterizedNodeEmbedding
from virtual_sensing.models.assets.my_graph_conv import myGraphConv
# mlp
from tsl.nn.blocks import MLP
from virtual_sensing.models.assets.multi_mlp import MultiMLP


def cat_all(*args):
    all = []
    for arg in args:
        if arg is not None:
            all.append(arg) 
    inputs = torch.cat(all, dim=-1)  
    return inputs


class GgNetTemporalConv(nn.Module):
    """
    Initializes the temporal convolution in a TGg block.

    Parameters:
        input_size (int): Input dimension.
        output_size (int): Output dimension.
        n_instances (int): Number of channels.
        n_tconv_layers (int): The number of layers for temporal convolution. Defaults to 3.
        kernel_size (int): The size of the convolutional kernel.
        dropout (float): Dropout probability. Defaults to 0.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_instances: int,
                 n_tconv_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.):
        super(GgNetTemporalConv, self).__init__()

        self.temporal_convs = nn.ModuleList([MultiTemporalConv(input_channels = input_size if l==0 else output_size,
                                                               output_channels = output_size,
                                                               kernel_size = kernel_size,
                                                               n_instances = n_instances,
                                                               dilation = 2**l,
                                                               stride = 1)
                                            for l in range(n_tconv_layers)])
    
        # assert kernel size is odd, else print error
        if kernel_size % 2 == 0: raise AssertionError('Kernel size must be odd, so that padding is symmetric.')
        for l in range(n_tconv_layers):
            self.temporal_convs[l]._causal_pad_sizes = (0, 0, 0, 0, ((kernel_size-1)//2)*(2**l), ((kernel_size-1)//2)*(2**l))
    
        self.dropout = nn.Dropout(dropout)
        self.norm_layer = LayerNorm(output_size)


    def forward(self,
                h: Tensor):
        """ temporal conv """
        # b t n f h -> b t n f h
        for tconv in self.temporal_convs:
            h = tconv(h)
            h = self.dropout(h)
        h = self.norm_layer(h)
        return h


class TGgBlock(nn.Module):
    """
    Initializes the TGgBlock module.

    Parameters:
        hidden_size (int): The number of hidden features or channels.
        embedding_G_size (int): The size of the embeddings for the G graph.
        embedding_g_size (int): The size of the embeddings for the g graph.
        exog_size (int): The size of exogenous input features.
        cat_emb_G (bool): Whether to concatenate G node embeddings
        cat_emb_g (bool): Whether to concatenate g node embeddings
        n_tconv_layers (int): The number of layers for temporal convolution.
        kernel_size (int): The size of the convolutional kernel.
        Gconv_mode (str): The mode for G graph convolution. Options: 'single' or 'multi'.
        dropout (float): Dropout probability.
        residual (bool): Whether to use residual connections.
        mlps (bool): Whether to add dense layers.
        l_norms (bool): Whether to use layer normalization.
        input_size (int): The number of input channels.
    """
    def __init__(self,
                 hidden_size,
                 embedding_G_size,
                 embedding_g_size,
                 exog_size,
                 cat_emb_G,
                 cat_emb_g,
                 n_tconv_layers,
                 kernel_size,
                 Gconv_mode,
                 dropout,
                 residual,
                 mlps,
                 l_norms,
                 input_size):
        super(TGgBlock, self).__init__()
        self.cat_emb_G = cat_emb_G
        self.cat_emb_g = cat_emb_g
        self.residual = residual
        self.mlps = mlps
        self.l_norms = l_norms

        tconv_in_size = hidden_size + 1 + exog_size      # hidden + mask + exog
        Gconv_in_size = hidden_size + 1 + exog_size
        gconv_in_size = hidden_size + 1 + exog_size

        # decide which layers get concatenated embeddings
        if cat_emb_G: 
            tconv_in_size += embedding_G_size
            Gconv_in_size += embedding_G_size
            gconv_in_size += embedding_G_size
        if cat_emb_g: 
            # tconv are already "multi"
            Gconv_in_size += embedding_g_size
            gconv_in_size += embedding_g_size

        #### Temporal conv ####
        self.tconv = GgNetTemporalConv(input_size = tconv_in_size,
                                       output_size = hidden_size,
                                       n_instances = input_size,
                                       n_tconv_layers = n_tconv_layers,
                                       kernel_size = kernel_size,
                                       dropout=0.) 

        #### Outer level G conv ####
        self.Gconv_mode = Gconv_mode
        self.Gconv = myGraphConv(input_size=Gconv_in_size,
                                 output_size=hidden_size,
                                 bias=True,
                                 root_weight=True,
                                 root_weight_type='linear',
                                 activation='elu',
                                 mode=Gconv_mode,
                                 n_instances=input_size if Gconv_mode=='multi' else None,
                                 pattern='b t n f h', instance_dim='f')

        #### second Inner level g conv ####
        self.gconv = myGraphConv(input_size=gconv_in_size,
                                 output_size=hidden_size,
                                 bias=True,
                                 root_weight=True,
                                 root_weight_type='linear',
                                 activation='elu')
    
        self.g_dropout = nn.Dropout(dropout)

        if l_norms:
            self.g_lnorm = LayerNorm(hidden_size)

        if mlps: 
            self.T_dense = Dense(hidden_size, hidden_size, activation='elu')
            self.G_mlp = Dense(hidden_size, hidden_size, activation='elu')
            self.g_mlp = Dense(hidden_size, hidden_size, activation='elu')

        
    def forward(self, h, mask, u, embs_G, embs_g, adj_G, adj_g,
                nodes, features):
 
        """########################################  h = Tconv(h, m, eG)
        # T conv                                  #  
        ########################################"""   
        res = h     
        h = self.tconv(cat_all(h, mask, u, 
                               embs_G if self.cat_emb_G else None))
        if self.residual: h += res                # skip (provide Gconv with unsmoothed information)
        if self.mlps:     h = self.T_dense(h)     # optional dense (can be modified to parametrize skip connections)

        """########################################  h = MessagePassing_G(h, m, eG, eg, adj_G)
        # G conv (inter-node)                     #  
        ########################################""" 
        if self.Gconv_mode == 'single':
            h, mask, embs_G, embs_g = GGNetModel.reshape_all('(b n) t f h', '(b f) t n h', nodes=nodes, args=[h, mask, embs_G, embs_g])
        elif self.Gconv_mode == 'multi':
            h, mask, embs_G, embs_g = GGNetModel.reshape_all('(b n) t f h', 'b t n f h', nodes=nodes, args=[h, mask, embs_G, embs_g])
        
        h_T = h
        h = self.Gconv(cat_all(h, mask, u, 
                               embs_G if self.cat_emb_G else None, 
                               embs_g if self.cat_emb_g else None), 
                       adj_G)   
        if self.residual: h += h_T             # skip (provide g-conv with same input as G-conv)
        if self.mlps:     h = self.G_mlp(h)    # optional dense (can be modified to parametrize skip connections)

        if self.Gconv_mode == 'single':
            h, mask, embs_G, embs_g = GGNetModel.reshape_all('(b f) t n h', '(b n) t f h', features=features, args=[h, mask, embs_G, embs_g])
        elif self.Gconv_mode == 'multi':
            h, mask, embs_G, embs_g = GGNetModel.reshape_all('b t n f h', '(b n) t f h', args=[h, mask, embs_G, embs_g])

        """########################################  h = MessagePassing_g(h, m, eG, eg, adj)
        # g conv (intra-node)                     #  
        ########################################"""
        h = self.gconv(cat_all(h, mask, u, 
                               embs_G if self.cat_emb_G else None, 
                               embs_g if self.cat_emb_g else None), 
                       adj_g)         
        h = self.g_dropout(h)                   # dropout
        if self.l_norms:  h = self.g_lnorm(h)   # layer norm
        if self.residual: h += res              # skip (provide next layer with unsmoothed information)
        if self.mlps:     h = self.g_mlp(h)     # optional dense (can be modified to parametrize skip connections)
        
        return h


class GGNetModel(BaseModel):
    """
    Initializes the GGNet module.

    Parameters:
        input_size (int): The number of input features or channels.
        hidden_size (int): The number of hidden features or channels.
        ff_size (int): The number of features or channels in the last feedforward layers.
        embedding_G_size (int): The size of the embeddings for the outer-level graph.
        embedding_G_h_size (int): The size of the embeddings for the outer-level graph.
        embedding_g_size (int): The size of the embeddings for the inner-level graph.
        exog_size (int): The size of exogenous input features.
        n_TGgconv_layers (int): The number of TGg convolutions.
        n_tconv_layers (int): The number of layers for temporal convolution.
        kernel_size (int): The size of the convolutional kernel.
        Gconv_mode (str): The mode for outer-level graph convolution. Options: 'single' or 'multi'.
        dropout (float): Dropout probability.
        cat_emb_G (bool): Whether to concatenate embeddings
        cat_emb_g (bool): Whether to concatenate embeddings
        n_clusters (int): The number of clusters for the embeddings.
        n_nodes (int): The number of nodes in the graph.
        multi_encoder (bool): Whether to use a multi-encoder.
        residual (bool): Whether to use residual connections.
        mlps (bool): Whether to use MLPs.
        l_norms (bool): Whether to use layer normalization.
        pinball (bool): Whether to use pinball loss.
    """

    return_type = list

    def __init__(self,
                input_size: int,
                hidden_size: int = 64,
                ff_size: int = 64,
                embedding_G_size: int = 16,
                embedding_G_h_size: int = 16,
                embedding_g_size: int = 8,
                exog_size: int = 0,

                n_TGgconv_layers: int = 2,
                n_MLPencoder_layers: int = 1,
                # tconv
                n_tconv_layers: int = 3,
                kernel_size: int = 3,
                # Gconv
                Gconv_mode: str = 'single',
                dropout: float = 0.,
                # embeddings
                cat_emb_G: bool = True,
                cat_emb_g: bool = True,
                n_clusters: int = 1,
                n_nodes = None,
                # others
                multi_encoder: bool = False,
                residual: bool = True,
                mlps: bool = False,
                l_norms: bool = True,
                # pinball
                pinball: bool = True):
    
        super(GGNetModel, self).__init__()

        self.cat_emb_G = cat_emb_G
        self.cat_emb_g = cat_emb_g
        self.multi_encoder = multi_encoder
        self.residual = residual
        self.l_norms = l_norms
        self.pinball = pinball

        #### Adj matrix #### 
        self.adj_g = nn.Parameter(Tensor(input_size, input_size), requires_grad=True)
        self.adj_g.data.uniform_(0, 1)
        # diagonal masks
        self.diag_mask_G = torch.zeros(n_nodes, n_nodes).fill_diagonal_((float('inf')))
        self.diag_mask_g = torch.zeros(input_size, input_size).fill_diagonal_((float('inf')))
        if torch.cuda.is_available():  
            self.diag_mask_G = self.diag_mask_G.to('cuda')
            self.diag_mask_g = self.diag_mask_g.to('cuda')

        #### Node embeddings #### 
        self.emb_G = ClusterizedNodeEmbedding(n_nodes, embedding_G_size, 
                                              n_clusters=n_clusters, tau=1.0,
                                              separation_loss=False, sep_eps=1.)
        self.emb_g = ClusterizedNodeEmbedding(input_size, embedding_g_size, 
                                              n_clusters=n_clusters, tau=1.0,
                                              separation_loss=False, sep_eps=1.)
        self.emb_G.init_centroids()
        self.emb_g.init_centroids()

        # MLP for location node embeddings
        self.emb_mlp_1 = MLP(embedding_G_size, 
                             embedding_G_h_size,
                             embedding_G_size, 
                             activation='tanh',
                             n_layers=1)
        self.emb_mlp_2 = MLP(embedding_G_size, 
                             embedding_G_h_size,
                             embedding_G_size, 
                             activation='tanh',
                             n_layers=1)
        
        #### Masked encoder ####
        self.emb_enc = MLP(exog_size+embedding_G_size+embedding_g_size, hidden_size, 
                            activation='elu', n_layers=n_MLPencoder_layers)
        if multi_encoder:
            self.x_enc = MultiMLP(1, hidden_size, 
                                  n_instances=input_size, 
                                  pattern='b t f h', instance_dim='f', 
                                  activation='elu', n_layers=n_MLPencoder_layers)
        else:
            self.x_enc = MLP(1, hidden_size, 
                             activation='elu', n_layers=n_MLPencoder_layers)

        if self.l_norms: self.norm_layer = LayerNorm(hidden_size)

        #### TGg convs ####
        self.TGg_convs = nn.ModuleList([TGgBlock(hidden_size,
                                                 embedding_G_size,
                                                 embedding_g_size,
                                                 exog_size,
                                                 cat_emb_G,
                                                 cat_emb_g,
                                                 n_tconv_layers,
                                                 kernel_size,
                                                 Gconv_mode,
                                                 dropout,
                                                 residual,
                                                 mlps,
                                                 l_norms,
                                                 input_size)
                                        for _ in range(n_TGgconv_layers)])

        #### MLP readout ####
        readout_in_size = hidden_size + 1   # hidden + mask
        if cat_emb_G: readout_in_size += embedding_G_size
        if cat_emb_g: readout_in_size += embedding_g_size
        self.mlp_out = MultiMLP(input_size = readout_in_size,
                                hidden_size = ff_size,
                                n_instances = input_size,
                                output_size = 3 if pinball else 1,
                                pattern = 'b t n f h', instance_dim = 'f',
                                activation='elu',
                                n_layers=1)

    @staticmethod
    def reshape_all(pattern_in, pattern_out, nodes=None, features=None, times=None, args=[]):
        all = []
        for arg in args:
            if arg is not None:
                if '(b n)' in pattern_in:
                    all.append(rearrange(arg, f'{pattern_in} -> {pattern_out}', n=nodes))  
                elif '(b f)' in pattern_in:
                    all.append(rearrange(arg, f'{pattern_in} -> {pattern_out}', f=features))  
                elif '(b t)' in pattern_in:
                    all.append(rearrange(arg, f'{pattern_in} -> {pattern_out}', t=times))
                else:
                    all.append(rearrange(arg, f'{pattern_in} -> {pattern_out}'))
            else:
                all.append(None)
        return all


    def forward(self,
                x: Tensor,
                mask: OptTensor = None,
                u: OptTensor = None) -> list:
        """"""
        # x: [batch, time, nodes, features]
        batches, steps, nodes, features = x.size()

        # node indices
        emb_G = self.emb_G()
        emb_g = self.emb_g()

        # G adjacency
        emb_G_encoded_1 = self.emb_mlp_1(emb_G)
        emb_G_encoded_2 = self.emb_mlp_2(emb_G)
        adj_G = (emb_G_encoded_1 @ emb_G_encoded_2.T) - self.diag_mask_G[:nodes, :nodes]  # calculate the adjacency matrix
        adj_G = F.softmax(adj_G, dim=1)       # normalize with softmax

        # g adjacency
        adj_g = self.adj_g - self.diag_mask_g # -inf in diagonal
        adj_g = F.softmax(adj_g, dim=1)       # normalize with softmax

        """########################################
        # Reshape into nested graph               # x = [b t n f] -> [b t n f 1]
        ########################################"""        
        # Whiten missing values
        x = torch.where(mask, x, torch.zeros_like(x))
        if torch.isnan(x).any(): raise ValueError('some NaNs are non masked in input data')
        # reshaping
        x, mask, u = GGNetModel.reshape_all('b t n f', '(b n) t f 1', args=[x, mask, u])
        embs_G = repeat(emb_G, f'n h -> (b n) t f h', b=batches, t=steps, f=features)
        embs_g = repeat(emb_g, f'f h -> (b n) t f h', b=batches, t=steps, n=nodes)


        """########################################  h = MLP(x) + MLP(e)
        # Masked encoder                          #  (b n) t f 1 -> (b n) t f h 
        ########################################"""  # Encode inputs, from embeddings where missing
        h_emb = self.emb_enc(torch.cat([embs_G, embs_g], axis=-1))
        h_data = self.x_enc(x) + h_emb
        h = torch.where(mask.bool(), h_data, h_emb)
        
        # normalize features
        if self.l_norms: h = self.norm_layer(h)

        """########################################  
        # TGg convs                               #  
        ########################################"""  
        res = h
        for TGg_conv in self.TGg_convs:
            h = TGg_conv(h, mask, u, embs_G, embs_g, adj_G, adj_g,
                            nodes, features)
        if self.residual: h += res

        """########################################  x = MLP(h, eG, eg, m)
        # MLP Readout                             #  b t n f h -> b t n f 3 (3 pinball components)
        ########################################"""
        h, mask, embs_G, embs_g = GGNetModel.reshape_all('(b n) t f h', 'b t n f h', nodes=nodes, args=[h, mask, embs_G, embs_g])
        h = cat_all(h, mask, u, embs_G if self.cat_emb_G else None, embs_g if self.cat_emb_g else None)
        imputation = self.mlp_out(h)
        imputation = rearrange(imputation, 'b t n f k -> b t n (f k)')
        if self.pinball:
            x_hat = list(torch.split(imputation, features, dim = -1))
        else:
            x_hat = [imputation]

        return [x_hat, []]


    def predict(self,
                x: Tensor,
                mask: OptTensor = None,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x = x, 
                            mask = mask, 
                            u = u)