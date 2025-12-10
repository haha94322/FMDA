import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import math
from nncore.nn.builder import build_norm_layer
from nncore.nn.blocks.transformer import FeedForwardNetwork, MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self,
                 dims,
                 ratio=4,
                 p=0.1,
                 pre_norm=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(EncoderLayer, self).__init__()

        self._dims = dims
        self._ratio = ratio
        self._p = p
        self._pre_norm = pre_norm

        # self.att = MultiHeadAttention(dims, heads=heads, p=p)
        self.att = simam_crossmodel()

        # self.ffn = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        # self.norm2 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, x, y, pe=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = v if pe is None else v + pe
            q = q.permute(0, 2, 1)
            d = self.att(q, y.unsqueeze(-1))
            if len(d.size()) == 4:
                d = d.sum(-1).permute(1, 0, 2)
            else:
                d = d.permute(0, 2, 1)
            x = x + d
            # d = self.norm2(x)
            # d = self.ffn(d)
            # x = x + d

        return x


class simam_crossmodel(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_crossmodel, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        # self.q_porj = nn.Linear(512, 512)

    def forward(self, x, query1):

        # q = self.q_porj(q)

        query1 = query1

        b, c, n = x.size()

        n = n - 1


        x_minus_mu_square = (x - x.mean(dim=[2], keepdim=True)).pow(2)


        y = x_minus_mu_square / \
            (4 * (x_minus_mu_square.sum(dim=[2], keepdim=True) / n + self.e_lambda)) \
            + 0.5

        y = 1/(query1.pow(2) + self.e_lambda) * y


        return x * self.activaton(y)

class fly_1d(nn.Module):
    """feature fusion"""
    def __init__(self, moduledim = 512):
        super(fly_1d, self).__init__()

        self.cross = simam_crossmodel()


        self.Norm_1 = nn.BatchNorm1d(moduledim)
        self.Norm_2 = nn.BatchNorm1d(moduledim)
        self.Norm_3 = nn.BatchNorm1d(moduledim)


    def forward(self, value, query):

        # query = query.sum(1).unsqueeze(-2)
        size = query.size()

        value_new = self.cross(value, query)   #将跨模态信息映射于相同的语义空间内，使得通道能够予语义对齐。

        if len(size) == 4:
            value_new = value_new.sum(-1).permute(1, 2, 0)
            query = query.squeeze(-1).permute(1, 2, 0)

        value_time = self.Norm_1(value_new).permute(0, 2, 1)
        query_time = query.permute(0, 2, 1)
        # # value_new = value
        #
        # bs = value.size(0)
        # # 映射到频域，对通道进行感知
        value_fft_shift = torch.fft.fft(value_time, dim=-1)
        # value_fft_shift = torch.fft.fftshift(value_fft)
        query_fft_shift = torch.fft.fft(query_time, dim=-1)

        # query_fft_shift = torch.fft.fftshift(query_fft)

        value_real, value_imag = torch.abs(value_fft_shift), torch.angle(value_fft_shift) / math.pi * 180
        query_real, query_imag = torch.abs(query_fft_shift), torch.angle(query_fft_shift) / math.pi * 180
        #
        #
        phase_difference = torch.abs(value_imag - query_imag)
        amplitude_difference = torch.abs(value_real - query_real)
        weight = torch.exp(-(phase_difference + amplitude_difference))
        #
        if len(size) == 4:
            value_filter = query_fft_shift * (
                    1 - weight) + value_fft_shift * weight
        else:
            value_filter = query_fft_shift.repeat(1, value_fft_shift.size(1), 1) * (
                        1 - weight) + value_fft_shift * weight

        # # value_filter = value_fft_shift# * query_fft_shift.unsqueeze(1).repeat(1, value_fft_shift.size(1), 1)
        #
        fre_xo = torch.abs(torch.fft.ifft(value_filter, dim=-1)) #获取频域增强信号_通过通道解偶获取的细节信息
        # fre_xo = self.Norm_1(fre_xo.permute(0, 2, 1)).permute(0, 2, 1)

        #
        """
        时域与频域之间的交叉互补
        """
        w_b_2_a = F.softmax(torch.bmm(query_time,fre_xo.permute(0, 2, 1)), dim=-1)
        w_a_2_b = F.softmax(torch.bmm(value_time, query), dim=-2)

        c = torch.bmm(w_b_2_a, fre_xo)
        d = torch.bmm(w_a_2_b, c)
        visual_time_out = value_time + d

        w_b_2_a = F.softmax(torch.bmm(query_time, value_time.permute(0, 2, 1)), dim=-1)
        w_a_2_b = F.softmax(torch.bmm(fre_xo, query), dim=-2)

        c = torch.bmm(w_b_2_a, value_time)
        d = torch.bmm(w_a_2_b, c)
        visual_fre_out = fre_xo + d

        visual = visual_time_out + visual_fre_out

        # xo = self.Norm_3(visual.permute(0, 2, 1)).permute(0, 2, 1)


        return visual





