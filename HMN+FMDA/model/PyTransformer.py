import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Function
from torch.nn import functional as F
from .modules.TransformerEncoders import *
from .modules.transformer import BottleneckTransformer


class ADA(nn.Module):
    def __init__(self, embed_dim):
        super(ADA, self).__init__()
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim ** -0.5
        self.activ=nn.GELU()

        self.proj1 = nn.Sequential(nn.Linear(embed_dim,embed_dim//2), self.activ)
        self.proj2 = nn.Sequential(nn.Linear(embed_dim,embed_dim//2), self.activ)

    def forward(self, v_1, v_2):
  
        v_1 = v_1.transpose(0,1)
        v_2 = v_2.transpose(0,1)
        attn = F.softmax(torch.bmm(self.proj1(v_1),self.proj2(v_2).transpose(1,2))*self.scaling,dim=-1) #Batch X1 X2
        v_1 = v_1 + torch.bmm(attn,v_2)
        return v_1.transpose(0,1)

class CrossModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 enc_cfg=None):
        super(CrossModalEncoder, self).__init__()


        # self.pos_enc = build_model(pos_cfg, dims)
        self.pos_enc = None
        self.encoder = BottleneckTransformer(dims)
        # self.mapping = build_linear_modules(map_dims, **kwargs)


    def forward(self, a, y, z):
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(a)
            a, b = self.encoder(a, y, z)

        return a, b


class deTransformer(nn.Module):
    def __init__(self, module_dim, num_layers=1):
        super().__init__()

        self.module_dim = module_dim

        self.num_layers = num_layers
        self.proj = nn.Linear(self.module_dim, self.module_dim)
        # self.q_proj = nn.Linear(self.module_dim, self.module_dim)
        self.p_proj = nn.Linear(self.module_dim, self.module_dim)


    def forward(self, a, q, bertbd):

        # visual_m_post = self.q_proj(bertbd)
        visual_m_post = bertbd
        # visual_a = self.p_proj(a[-1])
        # out = visual_a
        for index, layer in enumerate(range(0, self.num_layers - 1,  1)):
            visual_a = self.p_proj(a[layer + 1].permute(1, 0, 2))
            if index == 0:
                visual_a_post = self.proj(a[layer].permute(1, 0, 2))
                w_b_2_a = F.softmax(torch.bmm(visual_m_post,
                                              visual_a_post.permute(0, 2, 1)), dim=-1)
                w_a_2_b = F.softmax(torch.bmm(visual_a,
                                              visual_m_post.permute(0, 2, 1)), dim=-1)

                b = visual_a_post
                c = torch.bmm(w_b_2_a, b)
                d = torch.bmm(w_a_2_b, c)
                visual_a_post = visual_a + d
            else:
                w_b_2_a = F.softmax(torch.bmm(visual_m_post,
                                              visual_a_post.permute(0, 2, 1)), dim=-1)
                w_a_2_b = F.softmax(torch.bmm(visual_a,
                                              visual_m_post.permute(0, 2, 1)), dim=-1)

                b = visual_a_post
                c = torch.bmm(w_b_2_a, b)
                d = torch.bmm(w_a_2_b, c)
                visual_a_post = visual_a + d

        w_a_2_b = F.softmax(torch.bmm(q.unsqueeze(-2),
                                      visual_a_post.permute(0, 2, 1)), dim=-1)
        out = torch.bmm(w_a_2_b, visual_a_post)

        return out.squeeze(1)





class PyTransformer(nn.Module):
    def __init__(self, level, embed_dim, num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1):
        super(PyTransformer, self).__init__()
        self.level = level
        self.scaling = embed_dim ** -0.5
        self.activ=nn.GELU()

        self.CrossmodalEncoder = TransformerEncoder_nopos_v2(embed_dim, num_heads, attn_dropout, res_dropout,
                                                             activ_dropout)
        self.Up_ = nn.ModuleList([])
        for i in range(self.level-1):
            self.Up_.append(ADA(embed_dim))
        
        self.TransformerEncoder = deTransformer(embed_dim, level)
        # self.TransformerEncoder = TransformerEncoder_nopos_v1(embed_dim, num_heads, attn_dropout, res_dropout,
        #                                                       activ_dropout)

    def forward(self, visual_list, text_embedding, question_embedding_all, text_len):
  
        v_media = []
        for i in range(self.level):
            visual_list[i], text_embedding = self.CrossmodalEncoder(visual_list[i],
                                                                    text_embedding,
                                                                    question_embedding_all,
                                                                    text_len)

            # visual_list[i] = visual.permute(1, 0, 2)
            v_media.append(visual_list[i].clone())
            if i<self.level-1:
                visual_list[i+1] = self.Up_[i-1](visual_list[i+1], visual_list[i])

        Out = self.TransformerEncoder(v_media, question_embedding_all, text_embedding.permute(1, 0, 2))
        # for i in range(self.level):
        #     v_media[i] = self.TransformerEncoder(v_media[i],v_media[i],v_media[i])
        # # visual_out = v_media_out.unsqueeze(-2)
        # visual_out = torch.stack([item.mean(dim=0) for item in v_media], dim=1)
        # # text_out = question_embedding_all.unsqueeze(-1)
        # text_out = torch.stack([text_embedding[0:text_len[j], j, :].mean(dim=0) for j in range(text_len.shape[0])],
        #                        dim=0).unsqueeze(-1)
        # Out = (visual_out*F.softmax(torch.bmm(visual_out,text_out)*self.scaling,dim=1)).sum(dim=1)

        return Out