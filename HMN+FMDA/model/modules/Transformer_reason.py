import torch
import torch.nn as nn
from torch.nn import functional as F
from .TransformerEncoders import TransformerEncoder
import numpy as np




class enTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1,
                 activ_dropout=0.1, res_dropout=0.1, num_layers=1, num_tf_layers = 1, num_frames = 8):
        super().__init__()

        self.layers_a = nn.ModuleList([])
        self.layers_m = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer_a = TransformerEncoder(embed_dim, num_heads,
                                           attn_dropout, activ_dropout,
                                           res_dropout, num_tf_layers)
            new_layer_m = TransformerEncoder(embed_dim, num_heads,
                                           attn_dropout, activ_dropout,
                                           res_dropout, num_tf_layers)
            self.layers_a.append(new_layer_a)
            self.layers_m.append(new_layer_m)

        # self.layers_a = TransformerEncoder(embed_dim, num_heads,
        #                                  attn_dropout, activ_dropout,
        #                                  res_dropout, num_tf_layers)
        # self.layers_m = TransformerEncoder(embed_dim, num_heads,
        #                                  attn_dropout, activ_dropout,
        #                                  res_dropout, num_tf_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.num_layers = num_layers
        self.num_frames = num_frames

        self.pool_m = nn.ModuleList([])
        self.pool_a = nn.ModuleList([])
        for i in range(self.num_layers):
            self.pool_m.append(nn.AdaptiveAvgPool1d((self.num_frames // np.power(2, i))))
            self.pool_a.append(
                nn.AdaptiveMaxPool2d((int(np.ceil(7 / np.power(2, i))), int(np.ceil(7 / np.power(2, i))))))

    def forward(self, visual_list_fm, visual_list_fa, question_embedding, question_len):

        visual_out_a = []
        visual_out_m = []

        visual_pre_a = None
        visual_pre_m = None
        for layer in range(self.num_layers):

            w, h, l, bs = visual_list_fa[layer].size()

            if visual_pre_a == None:
                a = visual_list_fa[layer].permute(2, 3, 0, 1).flatten(-2).permute(2, 0, 1)
                m = visual_list_fm[layer]

            else:
                visual_pre_a = self.pool_a[layer](visual_pre_a)
                visual_pre_m = self.pool_m[layer](visual_pre_m)

                a = visual_list_fa[layer].permute(2, 3, 0, 1) + visual_pre_a
                a = a.flatten(-2).permute(2, 0, 1)

                m = visual_list_fm[layer] + visual_pre_m.permute(2, 0, 1)

            visual_a = self.layers_a[layer](a, question_embedding, question_embedding, question_len)
            visual_m = self.layers_m[layer](m, question_embedding, question_embedding, question_len)
            visual_pre_a = visual_a.permute(1, 2, 0).reshape(l, bs, w, h)
            visual_pre_m = visual_m.permute(1, 2, 0)

            visual_out_a.append(visual_a)
            visual_out_m.append(visual_m)

        return visual_out_a, visual_out_m

class deTransformer(nn.Module):
    def __init__(self, module_dim, num_layers=1, num_frames = 8):
        super().__init__()

        self.module_dim = module_dim

        self.num_layers = num_layers
        self.a_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.b_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.c_proj = nn.Linear(module_dim, module_dim, bias=False)


    def forward(self, a, m):

        out = []
        for layer in range(self.num_layers - 1):
            visual_a = self.a_proj(a[layer])
            visual_a_post = self.b_proj(a[layer + 1])
            visual_m_post = self.c_proj(m[layer + 1])

            w_b_2_a = F.softmax(torch.bmm(visual_m_post.permute(1, 0, 2),
                                          visual_a_post.permute(1, 2, 0)), dim=-1)
            w_a_2_b = F.softmax(torch.bmm(visual_a.permute(1, 0, 2),
                                          visual_m_post.permute(1, 2, 0)), dim=-1)

            b = a[layer + 1].permute(1, 0, 2)
            c = torch.bmm(w_b_2_a, b)
            d = torch.bmm(w_a_2_b, c)
            out.append(a[layer].permute(1, 0, 2) + d)

        out.append(a[-1].permute(1, 0, 2))

        return out


class endTransformer(nn.Module):
    def __init__(self, module_dim, num_layers=1, num_frames = 8):
        super().__init__()

        self.module_dim = module_dim
        self.num_layers = num_layers
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))



    def forward(self, a, m, q):
        out = []
        question = q.mean(1).unsqueeze(1)
        for layer in range(self.num_layers):
            visual_a = a[layer]
            visual_m = m[layer]
            alph = F.softmax(torch.bmm(question, visual_a.transpose(1, 2)).transpose(1, 2), dim=1)
            beta = F.softmax(torch.bmm(question, visual_m.transpose(1, 2)).transpose(1, 2), dim=1)
            out_l = self.a * (alph * visual_a).sum(1) + self.b * (beta * visual_m).sum(1)

            out.append(out_l)

        return out