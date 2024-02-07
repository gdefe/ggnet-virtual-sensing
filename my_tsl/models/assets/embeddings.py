import math
from typing import Optional, List

import torch
from sklearn.cluster import KMeans
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.typing import OptTensor
from tsl.nn.layers import NodeEmbedding


class VariationalNodeEmbedding(NodeEmbedding):

    def __init__(self, n_tokens: int, emb_size: int,
                 initial_var: float = 0.2):
        super().__init__(n_tokens, emb_size)
        self.log_var = nn.Parameter(Tensor(self.n_tokens, self.emb_size))
        self.initial_var = initial_var
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            bound = 1e-2
            self.emb.data.uniform_(-bound, bound)
            self.log_var.data.fill_(math.log(self.initial_var))

    def forward(self, expand: Optional[List] = None,
                node_index: OptTensor = None,
                tokens_first: bool = True):
        """"""
        if self.training:
            mu, std = self.emb, torch.exp(self.log_var / 2)
            if node_index is not None:
                mu, std = mu[node_index], std[node_index]
            if not tokens_first:
                mu, std = mu.T, std.T
            if expand is not None:
                shape = [*mu.size()]
                view = [1 if d > 0 else shape.pop(0 if tokens_first else -1)
                        for d in expand]
                mu = mu.view(*view).expand(*expand)
                std = std.view(*view).expand(*expand)
            return mu + std * torch.randn_like(std)
        else:
            return super().forward(expand, node_index, tokens_first)


class NormalizedNodeEmbedding(NodeEmbedding):
    def __init__(self, n_tokens: int, emb_size: int,
                 norm: str = None,
                 requires_grad: bool = True,
                 initializer: str = 'same'):
        assert initializer in ['same', 'uniform']
        assert isinstance(norm, int) or norm in [None, 'none', 'softmax']
        super().__init__(n_tokens, emb_size, initializer, requires_grad)
        self.norm = norm

    def reset_emb(self):
        with torch.no_grad():
            if self.initializer == 'same':
                self.emb.data.fill_(1.0 / self.emb.size(-1))
            elif self.initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.emb.size(-1))
                self.emb.data.uniform_(-bound, bound)
            else:
                raise RuntimeError(f"Embedding initializer '{self.initializer}'"
                                   " is not supported.")

    def get_emb(self):
        if self.norm is None or self.norm == 'none':
            return self.emb
        elif self.norm == 'softmax':
            return torch.softmax(self.emb, dim=-1)
        elif isinstance(self.norm, int):
            return (self.emb / torch.linalg.norm(self.emb, self.norm, dim=-1,
                                                 keepdims=True)) ** self.norm
        else:
            raise RuntimeError(f"Normalization '{self.norm}' is not supported.")


class FactorizedNodeEmbedding(NormalizedNodeEmbedding):
    def __init__(self, n_tokens: int, emb_size: int, n_basis: int,
                 norm: str = None, requires_grad: bool = True,
                 initializer: str = 'same'):
        super().__init__(n_tokens, n_basis, norm, requires_grad, initializer)
        self.n_basis = n_basis
        self.emb_size = emb_size
        self.basis = nn.Parameter(Tensor(self.n_basis, self.emb_size),
                                  requires_grad=requires_grad)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"n_tokens={self.n_tokens}, embedding_size={self.emb_size}, " \
               f"n_basis={self.n_basis}"

    def reset_parameters(self):
        self.reset_emb()
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.basis.size(-1))
            self.basis.data.uniform_(-bound, bound)

    def get_emb(self):
        emb = super().get_emb()
        return torch.matmul(emb, self.basis)


class ClusterizedNodeEmbedding(NodeEmbedding):
    def __init__(self, n_tokens: int, emb_size: int, n_clusters: int,
                 tau: int = 1.,
                 estimator: str = 'ste', learned_assignments: bool = True,
                 requires_grad: bool = True,
                 separation_loss: bool = False, sep_eps: float = 1.,
                 temp_annealing: bool = False,
                 temp_decay_coeff: float = 0.99
                 ):
        super().__init__(n_tokens, emb_size, requires_grad=requires_grad)
        self.n_clusters = n_clusters
        self.estimator = estimator
        self.learned_assignments = learned_assignments
        self._tau = tau
        self.temp_annealing = temp_annealing
        self.temp_decay_coeff = temp_decay_coeff
        self.separation_loss = separation_loss
        self.max_sep = sep_eps
        self._frozen_centroids = False

        self.emb = nn.Parameter(Tensor(n_tokens, self.emb_size),
                                requires_grad=requires_grad)
        self.centroids = nn.Parameter(Tensor(n_clusters, self.emb_size),
                                      requires_grad=requires_grad)
        if learned_assignments:
            self.cluster_assignment = nn.Parameter(Tensor(n_tokens, n_clusters),
                                                   requires_grad=requires_grad)
        else:
            self.register_parameter('cluster_assignment', None)

        self.reset_parameters()

    @property
    def tau(self):
        return self._tau

    def step(self):
        if self.temp_annealing:
            self._tau = max(self.temp_decay_coeff * self._tau, 0.0001)

    def extra_repr(self) -> str:
        return f"n_tokens={self.n_tokens}, embedding_size={self.emb_size}, " \
               f"n_clusters={self.n_clusters}"

    @torch.no_grad()
    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.emb.size(-1))
        self.emb.data.uniform_(-bound, bound)
        self.centroids.data.uniform_(-bound, bound)
        if self.cluster_assignment is not None:
            self.cluster_assignment.data.uniform_(0., 1.)

    def assignment_logits(self):
        if self.learned_assignments:
            return self.cluster_assignment
        return torch.cdist(self.emb, self.centroids)

    def assigment_entropy(self):
        soft_assignments = self.get_assignment(estimator='soft')
        entropy = (- torch.sum(soft_assignments * torch.log(soft_assignments),
                               -1)).mean()
        return entropy

    def get_assignment(self, estimator=None):
        if estimator is None:
            estimator = self.estimator
        logits = self.assignment_logits()

        if estimator == 'ste':
            soft_assignment = torch.softmax(logits / self.tau, -1)
            idx = torch.argmax(soft_assignment, -1)
            hard_assignment = F.one_hot(idx, num_classes=self.n_clusters)
            return hard_assignment + soft_assignment - soft_assignment.detach()
        elif estimator == 'gt':
            g = -torch.log(-torch.log(torch.rand_like(logits)))
            scores = logits / self.tau + g
            return torch.softmax(scores, -1)
        elif estimator == 'soft':
            return torch.softmax(logits / self.tau, -1)
        else:
            raise NotImplementedError(
                f'{self.estimator} is not a valid a trick.')

    def init_centroids(self):
        if not self._frozen_centroids:
            kmeans = KMeans(n_clusters=self.n_clusters)
            X = self.emb.detach().cpu().numpy()
            kmeans.fit(X)
            centroids = torch.tensor(kmeans.cluster_centers_,
                                     dtype=torch.float32)
            self.centroids.data.copy_(centroids)
        if self.learned_assignments:
            dist = torch.cdist(self.emb, self.centroids)
            self.cluster_assignment.data.copy_(dist)

    def freeze_centroids(self):
        self._frozen_centroids = True

    def unfreeze_centroids(self):
        self._frozen_centroids = False

    def clustering_loss(self, node_idx=None):
        assignment = self.get_assignment()
        node_centroid = torch.matmul(assignment, self.centroids)
        dist = torch.norm(self.emb - node_centroid, p=2, dim=-1)
        # if node_idx:
        #     dist = dist[node_idx]
        dist = dist.mean()
        if self.separation_loss:
            sep = torch.cdist(self.centroids, self.centroids)
            return dist - torch.minimum(sep, self.max_sep * torch.ones_like(
                sep)).mean(), dist
        return dist, dist
