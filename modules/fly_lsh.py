import torch
from torch import nn


class SparseBinaryConstraint(object):

    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data = module.weight.data * self.mask
            module.weight.data[torch.where(self.mask)] = 1.0

        if hasattr(module, 'bias'):
            module.bias.data = module.bias.data * 0.0


class kWTA(nn.Module):

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        k_inds = (-x).topk(x.shape[-1] - self.k, dim=-1).indices
        x = x.scatter_(-1, k_inds, 0.0)

        return x


class FlyLSH(nn.Module):

    def __init__(self, input_dim: int, out_dim: int, tag_dim: int, weight_sparsity: float = 0.3):
        """

        :param input_dim:
        :param out_dim:
        :param tag_dim:
        :param weight_sparsity:
        """

        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.tag_dim = tag_dim
        self.weight_sparsity = weight_sparsity

        self.num_projections = int(self.weight_sparsity * self.input_dim)

        # constructing the projection matrix mask
        row_indices = torch.randint(0, input_dim, size=(self.num_projections * self.out_dim,))
        col_indices = torch.arange(0, self.out_dim).repeat_interleave(self.num_projections)
        mask = torch.zeros((self.out_dim, self.input_dim)).long()
        mask[col_indices, row_indices] = 1.0
        self.constrainer = SparseBinaryConstraint(mask)

        # construct linear module
        self.projection = nn.Linear(input_dim, out_dim)
        self.projection.apply(self.constrainer)

        # winner take all
        self.wta = kWTA(k=self.tag_dim)

    def forward(self, x):
        # 1. center the mean
        x = x - torch.mean(x, dim=-1, keepdim=True).repeat_interleave(x.shape[-1], -1)

        # 2. project to KCs
        kc = self.projection(x)

        # 3. winner take all
        kc_tag = self.wta(kc)

        return kc_tag
