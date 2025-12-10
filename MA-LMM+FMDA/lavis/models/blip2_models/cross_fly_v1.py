import torch.nn as nn
import torch
import numpy as np
import math
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

# from .cross_fly_v16 import channel_crossmodel



class pearson(torch.nn.Module):
    def __init__(self, moduledim=512, channels=None, e_lambda=1e-4):
        super(pearson, self).__init__()

        self.activaton = nn.LeakyReLU()
        self.e_lambda = e_lambda
        # self.Norm = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.Norm2 = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.q_porj = nn.Linear(512, 512)

    def forward(self, x, y):
        mean_x = x.mean(dim=[-1], keepdim=True)
        mean_y = y.mean(dim=[-1], keepdim=True)

        # 计算 Pearson 相关系数的分子部分
        cov_xy = (x - mean_x) * (y - mean_y)

        # 计算 Pearson 相关系数的分母部分
        # a = torch.std(x, unbiased=True)
        a = (x - mean_x).pow(2).sum(dim=[-1], keepdim=True)
        # b = torch.std(y, unbiased=True)
        b = (y - mean_x).pow(2).sum(dim=[-1], keepdim=True)

        # 计算 Pearson 相关系数
        rho = cov_xy / torch.sqrt(a * b + 0.001)
        """
        上述结果计算出跨模态特征每一个通道之间的相互关联，接着需要分别比较幅度和相角之间的联系，以确定特征通道特征元素之间的相似度。
        其中幅度控制相似度大小，相角控制相似度的正相关或负相关
        """
        value_real = torch.abs(rho)
        # a = value_real.max()
        value_imag = torch.atan2(rho.imag, (rho.real + 0.0001))
        # value_real = self.Norm(value_real.permute(0, 2, 1)).permute(0, 2, 1)
        # value_imag = self.Norm2(value_imag.permute(0, 2, 1)).permute(0, 2, 1)
        # value_imag[torch.isnan(value_imag)] = 0.0
        # value_imag = self.activaton(value_imag)
        value_imag = 2 / (1+torch.exp(-2*(value_imag) * 0.5)) - 1

        rho = value_real * value_imag

        # b = rho.max()

        # rho = torch.sigmoid(rho)

        # rho = torch.tanh(rho)
        # print(rho.max())
        return rho

class pearson_channel(torch.nn.Module):
    def __init__(self, moduledim=512, channels=None, e_lambda=1e-4):
        super(pearson_channel, self).__init__()

        self.activaton = nn.LeakyReLU()
        self.e_lambda = e_lambda
        # self.Norm = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.Norm2 = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.q_porj = nn.Linear(512, 512)

    def pearson_distance(self, x, y):
        mean_x = x.mean(dim=[-1], keepdim=True)
        mean_y = y.mean(dim=[-1], keepdim=True)
        c = torch.tensor([True]).cuda()
        d = torch.tensor([0]).cuda()
        # 计算 Pearson 相关系数的分子部分
        cov_xy = (x - mean_x) * (y - mean_y)

        # 计算 Pearson 相关系数的分母部分
        # a = torch.std(x, unbiased=True)
        a = (x - mean_x).pow(2).sum(dim=[-1], keepdim=True)

        # b = torch.std(y, unbiased=True)
        b = (y - mean_y).pow(2).sum(dim=[-1], keepdim=True)
        # print(torch.abs(b).max())
        # 计算 Pearson 相关系数
        rho = cov_xy / (torch.sqrt(a * b + 0.0001))

        return rho

    def forward(self, x, y):

        matrix_tmp = self.pearson_distance(x, y)  # ** 2
        value_imag = torch.atan2(matrix_tmp.imag, (matrix_tmp.real + 0.0001))
        weight_matrix = 2 / (1 + torch.exp(-2 * (value_imag) * 0.15)) - 1
        weight_matrix[torch.isnan(weight_matrix)] = 0.0
        weight_matrix = weight_matrix.clone().detach()

        # print(weight_matrix.max(),weight_matrix.min())

        freq_distance = (torch.abs(x - y)) ** 2
        weight = torch.exp(-(torch.abs(freq_distance)) * 0.5)
        # print(weight.max(), weight.min())
        weight = weight * weight_matrix
        # print(weight.max(), weight.min(), weight.mean())
        return weight

def loss_formulation(freqs, alpha = 1.0):


    freqs = torch.cat(
            [value_fft.unsqueeze(0) for value_fft in freqs],
            dim=0)

    num, bs, h, c = freqs.size()

    freqs = torch.fft.fft(freqs, dim=-1, norm='forward')

    freqs_a = freqs.unsqueeze(0).repeat(len(freqs), 1, 1, 1, 1)
    freqs_b = freqs.unsqueeze(1).repeat(1, len(freqs), 1, 1, 1)

    diag = torch.eye(len(freqs)).cuda().repeat(bs, h, c, 1, 1).permute(3,4,0,1,2) * freqs_a

    freqs = torch.triu(freqs_a.permute(2,3,4,0,1)) + torch.tril(freqs_b.permute(2,3,4,0,1)) - diag.permute(2,3,4,0,1)

    freqs = freqs.permute(3,4,0,1,2)

    loss_bak = 0

    for i in range(len(freqs) - 1):
        recon_freq = freqs[i]
        real_freq = freqs[i+1]
        matrix_tmp = (torch.abs(recon_freq - real_freq))  # ** 2
        matrix_tmp = matrix_tmp / matrix_tmp.max()
        matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
        matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
        weight_matrix = matrix_tmp.clone().detach()
        tmp = (torch.abs(recon_freq - real_freq))  # ** 2
        # tmp = pearson(recon_freq, real_freq)
        freq_distance = tmp
        loss = weight_matrix * freq_distance
        loss_bak = loss_bak + loss


    loss = torch.mean(loss_bak)
    loss = torch.exp(-(torch.abs(loss)) * 0.5)  #目的是为了使得0处损失最大，两端最小。  使得特征尽可能的不同。
    # loss = y_values / y_values.max()
    return loss



class channel_crossmodel(torch.nn.Module):
    def __init__(self, moduledim, kernel_size, dsl_init_sigma, dsl_init_center, channels=None, e_lambda=1e-4):
        super(channel_crossmodel, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        self.kernel_num = 4

        self.sigmas = nn.Parameter(torch.empty(self.kernel_num, 1), requires_grad=True)
        self.center_shift = nn.Parameter(torch.empty(self.kernel_num, 1),
            requires_grad=True)
        nn.init.uniform_(self.sigmas)
        nn.init.uniform_(self.center_shift)

        self.Norm = nn.BatchNorm2d(768)
        self.pearson = pearson()

        self.kernel_size = kernel_size
        # print(self.kernel_size)
        self.padding = [self.kernel_size // 2, self.kernel_size // 2]
        # self.q_porj = nn.Linear(512, 512)

        self.calculate_loss = True

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)

        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def _get_gaussian_kernel1d(self, kernel_size, sigma, center_shift):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).cuda()
        pdf = torch.exp(-0.5 * ((x - center_shift) / (sigma + 0.001)).pow(2))
        kernel1d = pdf / (pdf.sum() + 0.01)
        return kernel1d


    def _gaussian_blur(self, x, i):
        x = F.pad(x, self.padding, mode="replicate")
        self.gauss_kernel = self._get_gaussian_kernel1d(self.kernel_size, self.sigmas[i], self.center_shift[i])
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-2], x.shape[-2], 1)
        feat = F.conv1d(x, self.gauss_kernel)
        return feat.to(torch.float32)

    def forward(self, x, q):
        # q = self.q_porj(q)
        q_bak = q.mean(-2)
        outputs = []
        x_shift_bak = []


        for i in range(self.kernel_num):
            x_shift = self._gaussian_blur(x, i)
            x_shift_bak.append(x_shift)

        x_shift = torch.cat(
            [value_fft.unsqueeze(1) for value_fft in x_shift_bak],
            dim=1)


        query_fft_shift = torch.fft.fft(q, dim=-2).unsqueeze(1)  # 不能随意修改形状，会造成梯度混乱。

        value_fft_shift = torch.fft.fft(x_shift, n=q.size(-2), dim=-2)

        weight = self.pearson(value_fft_shift, query_fft_shift)  # ** 2

        # value_filter = query_fft_shift * (1 - weight) + value_fft_shift * weight
        value_filter = value_fft_shift * weight

        outputs = torch.abs(torch.fft.ifft(value_filter, n=x.size(-2), dim=-2))

        outputs = self.Norm(outputs.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        w_b_2_a = F.softmax(torch.matmul(q_bak.unsqueeze(-2).repeat(1, outputs.size(1), 1).unsqueeze(-2),
                                         outputs.permute(0, 1, 3, 2)), dim=-1)

        outputs = torch.matmul(w_b_2_a.permute(0, 3, 2, 1), outputs.permute(0, 2, 1, 3)).squeeze()


        if self.calculate_loss and self.kernel_num != 1:
            loss = loss_formulation(x_shift_bak)
            # print(loss)
            # loss = torch.exp(-(torch.abs(loss)*0.01))
        else:
            loss = torch.tensor(0.0)

        return outputs, loss


class fly_1d(nn.Module):
    """feature fusion"""
    def __init__(self, kernel_size = 3, moduledim = 512, dsl_init_sigma = 3.0, dsl_init_center = 0.0):
        super(fly_1d, self).__init__()

        self.Norm_2 = nn.BatchNorm1d(moduledim)
        self.cross = channel_crossmodel(moduledim, kernel_size, dsl_init_sigma, dsl_init_center)

    def forward(self, bert_output, question_llm):
        value, query = bert_output, question_llm
        # print(value.max(), value.min(), value.mean())
        # print(video_query.max(), video_query.min(), video_query.mean())
        value, loss = self.cross(value, query)
        return value
