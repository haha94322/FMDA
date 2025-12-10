import torch.nn as nn
import torch
import numpy as np
import math
from torch.nn import functional as F


class fly_1d(nn.Module):
    """feature fusion"""
    def __init__(self, moduledim = 512):
        super(fly_1d, self).__init__()

        self.crossmodel = nn.MultiheadAttention(moduledim, 8, dropout=0.1)

        self.Norm_1 = nn.BatchNorm1d(moduledim)
        self.Norm_2 = nn.BatchNorm1d(moduledim)


    def forward(self, value, query):

        batch_size = value.size(0)
        srcs = value.permute(1, 0, 2)

        crn_feat_out = []
        q = query.unsqueeze(0).repeat(srcs.size(0), 1, 1)

        crn_feats, weight = self.crossmodel(q, srcs, value=srcs)

        crn_feats = self.Norm_1(crn_feats.permute(1, 2, 0)).permute(0, 2, 1)

        return crn_feats
