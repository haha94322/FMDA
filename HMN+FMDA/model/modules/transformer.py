# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch.nn as nn
from nncore.nn import (MODELS, FeedForwardNetwork, MultiHeadAttention,
                       Parameter, build_norm_layer)
from ..CRN import fly_1d as crossmodel


@MODELS.register()
class BottleneckTransformerLayer(nn.Module):

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(BottleneckTransformerLayer, self).__init__()

        self.dims = dims
        self.heads = heads
        self.ratio = ratio
        self.p = p

        # self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        # self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att1 = crossmodel()
        self.att2 = crossmodel()
        self.att3 = crossmodel()
        self.att4 = crossmodel()

        self.ffn1 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)
        self.norm4 = build_norm_layer(norm_cfg, dims=dims)
        self.norm5 = build_norm_layer(norm_cfg, dims=dims)
        self.norm6 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, a, t, z, pe=None):
        da = self.norm1(a)
        dt = self.norm3(z).permute(0, 2, 1)

        ka = da if pe is None else da + pe


        at = self.att1(dt, ka.permute(1, 0, 2).unsqueeze(-1)).sum(1)

        t = t + at
        dt = self.norm4(t).unsqueeze(1)

        qa = da if pe is None else da + pe


        a = a + self.att3(qa.permute(0, 2, 1), dt.permute(0, 2, 1))


        da = self.norm5(a)


        a = a + self.ffn1(da)


        return a, t


@MODELS.register()
class BottleneckTransformer(nn.Module):

    def __init__(self, dims, num_tokens=4, num_layers=1, **kwargs):
        super(BottleneckTransformer, self).__init__()

        self.dims = dims
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.encoder = nn.ModuleList([
            BottleneckTransformerLayer(dims, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, a, y, z):
        for enc in self.encoder:
            a, t = enc(a, y, z)
        return a, t
