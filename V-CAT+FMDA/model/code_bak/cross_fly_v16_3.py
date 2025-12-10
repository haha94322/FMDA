import torch.nn as nn
import torch
import numpy as np
import math
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

class pearson(torch.nn.Module):
    def __init__(self, moduledim=512, channels=None, e_lambda=1e-4):
        super(pearson, self).__init__()

        self.activaton = nn.LeakyReLU()
        self.e_lambda = e_lambda
        # self.Norm = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.Norm2 = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.q_porj = nn.Linear(512, 512)

    def forward(self, x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        # 计算 Pearson 相关系数的分子部分
        cov_xy = (x - mean_x) * (y - mean_y)

        # 计算 Pearson 相关系数的分母部分
        # a = torch.std(x, unbiased=True)
        a = (x - mean_x).pow(2).sum(dim=[-1], keepdim=True)
        # b = torch.std(y, unbiased=True)
        b = (y - mean_x).pow(2).sum(dim=[-1], keepdim=True)

        # 计算 Pearson 相关系数
        rho = cov_xy / (a * b + 0.001)
        """
        上述结果计算出跨模态特征每一个通道之间的相互关联，接着需要分别比较幅度和相角之间的联系，以确定特征通道特征元素之间的相似度。
        其中幅度控制相似度大小，相角控制相似度的正相关或负相关
        """
        value_real = torch.abs(rho)
        value_imag = torch.atan2(rho.imag, (rho.real + 0.0001))
        # value_real = self.Norm(value_real.permute(0, 2, 1)).permute(0, 2, 1)
        # value_imag = self.Norm2(value_imag.permute(0, 2, 1)).permute(0, 2, 1)
        # value_imag[torch.isnan(value_imag)] = 0.0
        # value_imag = self.activaton(value_imag)
        value_imag = 2 / (1+torch.exp(-2*(value_imag) * 0.5)) - 1

        rho = value_real * value_imag
        # rho = torch.tanh(rho)
        # print(rho.max())
        return rho



def loss_formulation(freqs, alpha = 1.0):

    freqs = torch.cat(
            [value_fft.unsqueeze(0) for value_fft in freqs],
            dim=0)

    freqs = torch.fft.fft(freqs, dim=-1, norm='forward')

    loss_bak = 0
    i_bak = []

    for i in range(len(freqs)):
        for j in range(len(freqs)):
            if i == j or j in i_bak:
                continue
            else:
                recon_freq = freqs[i]
                real_freq = freqs[j]

                matrix_tmp = (torch.abs(recon_freq - real_freq))# ** 2
                matrix_tmp = matrix_tmp / matrix_tmp.max()
                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()
                tmp = (torch.abs(recon_freq - real_freq))# ** 2
                # tmp = pearson(recon_freq, real_freq)
                freq_distance = tmp
                loss = weight_matrix * freq_distance
                loss_bak = loss_bak + loss

        i_bak.append(i)

    loss = torch.mean(loss_bak)
    loss = torch.exp(-(torch.abs(loss)) * 0.5)  #目的是为了使得0处损失最大，两端最小。  使得特征尽可能的不同。
    # loss = y_values / y_values.max()
    return loss




class simam_crossmodel(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_crossmodel, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        # self.q_porj = nn.Linear(512, 512)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)

        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x, q):

        # q = self.q_porj(q)

        q = q.unsqueeze(-1)

        b, c, n = x.size()

        n = n - 1

        x_minus_mu_square = (x - x.mean(dim=[2], keepdim=True)).pow(2)

        y = x_minus_mu_square / \
            (4 * (x_minus_mu_square.sum(dim=[2], keepdim=True) / n + self.e_lambda)) \
            + 0.5

        y = 1 / (q.pow(2) + self.e_lambda) * y

        # y = torch.bmm(a, y)

        return x * self.activaton(y)


class channel_crossmodel(torch.nn.Module):
    def __init__(self, moduledim, kernel_size, dsl_init_sigma, dsl_init_center, calculate_loss = False, channels=None, e_lambda=1e-4):
        super(channel_crossmodel, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.calculate_loss = calculate_loss
        self.kernel_num = 4

        self.sigmas = nn.Parameter(torch.empty(self.kernel_num, 1), requires_grad=True)
        self.center_shift = nn.Parameter(torch.empty(self.kernel_num, 1),
            requires_grad=True)

        nn.init.uniform_(self.sigmas)
        nn.init.uniform_(self.center_shift)

        self.kernel_size = kernel_size
        self.padding = [self.kernel_size // 2, self.kernel_size // 2]

        self.Norm = nn.BatchNorm2d(moduledim)
        self.pearson = pearson()
        # self.q_porj = nn.Linear(512, 512)

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
        return feat

    def forward(self, x, q):

        # q = self.q_porj(q)
        q_bak = q
        outputs = []
        x_shift_bak = []

        query_fft_shift = torch.fft.fft(q, dim=-1).unsqueeze(1).unsqueeze(1)
        # query_real, query_imag = torch.abs(query_fft_shift), torch.angle(query_fft_shift)# / math.pi * 180
        x_bak = torch.fft.fft(x, dim=-1).unsqueeze(-2).repeat(1, 1, self.kernel_num, 1)
        for i in range(self.kernel_num):
            x_shift = self._gaussian_blur(x, i)
            x_shift_bak.append(x_shift)

        x_shift = torch.cat(
            [value_fft.unsqueeze(-2) for value_fft in x_shift_bak],
            dim=-2)
        value_fft_shift = torch.fft.fft(x_shift, dim=-1)
        weight = self.pearson(value_fft_shift, query_fft_shift)  # ** 2
        value_filter = query_fft_shift.repeat(1,  value_fft_shift.size(1), value_fft_shift.size(2), 1) * (
                1 - weight) + x_bak * weight
        outputs = torch.abs(torch.fft.ifft(value_filter, dim=-1)) + x.unsqueeze(-2).repeat(1, 1, self.kernel_num, 1)


        if len(outputs.size()) == 5:
            outputs = outputs.mean(0)
            q_bak = q_bak.mean(0)

        outputs = self.Norm(outputs.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        w_b_2_a = F.softmax(torch.matmul(q_bak.unsqueeze(-2).repeat(1, outputs.size(1), 1).unsqueeze(-2),
                                         outputs.permute(0, 1, 3, 2)), dim=-1)

        outputs = torch.matmul(w_b_2_a, outputs)


        if self.calculate_loss:
            loss = loss_formulation(x_shift_bak)
            # print(loss)
            # loss = torch.exp(-(torch.abs(loss)*0.01))
        else:
            loss = torch.tensor(0.0)


        return outputs.squeeze(), loss

class fly_1d(nn.Module):
    """feature fusion"""
    def __init__(self, kernel_size = 3, moduledim = 512, calculate_loss = True, dsl_init_sigma = 3.0, dsl_init_center = 0.0):
        super(fly_1d, self).__init__()

        self.Norm_2 = nn.BatchNorm1d(moduledim)
        self.Norm_3 = nn.BatchNorm1d(moduledim)

        self.cross = simam_crossmodel()
        self.channel_cross = channel_crossmodel(moduledim, kernel_size, dsl_init_sigma, dsl_init_center, calculate_loss)

    def forward(self, value, query, text):

        # value_time = self.cross(value.permute(0, 2, 1), query)
        value_time, loss1 = value, 0.0
        # value_time = self.Norm_2(value_time).permute(0, 2, 1)
        value_channel, loss = self.channel_cross(value_time, query)
        # value_channel, loss2 = value, 0.0
        # value_channel = self.Norm_3(value_channel.permute(0, 2, 1)).permute(0, 2, 1)

        """
        # 时域与频域之间的交叉互补
        # """
        # w_b_2_a = F.softmax(torch.bmm(query.unsqueeze(-2), value_channel.permute(0, 2, 1)), dim=-1)
        # w_a_2_b = F.softmax(torch.bmm(value_time, query.unsqueeze(-1)), dim=-2)
        #
        # c = torch.bmm(w_b_2_a, value_channel)
        # d = torch.bmm(w_a_2_b, c)
        # visual_time_out = value_time + d
        #
        # w_b_2_a = F.softmax(torch.bmm(query.unsqueeze(-2), value_time.permute(0, 2, 1)), dim=-1)
        # w_a_2_b = F.softmax(torch.bmm(value_channel, query.unsqueeze(-1)), dim=-2)
        #
        # c = torch.bmm(w_b_2_a, value_time)
        # d = torch.bmm(w_a_2_b, c)
        # visual_fre_out = value_channel + d

        # visual = value_time + value_channel


        return value_channel, loss
