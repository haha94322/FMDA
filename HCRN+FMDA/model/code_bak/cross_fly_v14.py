import torch.nn as nn
import torch
import numpy as np
import math
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def pearson(x, y):
    # 计算均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 计算 Pearson 相关系数的分子部分
    cov_xy = (x - mean_x) * (y - mean_y)

    # 计算 Pearson 相关系数的分母部分
    # a = torch.std(x, unbiased=True)
    a = (x - mean_x).pow(2).sum(dim=[2], keepdim=True)
    # b = torch.std(y, unbiased=True)
    b = (y - mean_x).pow(2).sum(dim=[2], keepdim=True)

    # 计算 Pearson 相关系数
    rho = cov_xy / (a * b + 0.001)

    rho = 1 - torch.exp(-(torch.abs(rho) * 0.1))
    # rho = torch.abs(rho)
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
                # value_real, value_imag = torch.abs(freqs[i]), torch.angle(freqs[i])# / math.pi * 180
                # query_real, query_imag = torch.abs(freqs[j]), torch.angle(freqs[j])# / math.pi * 180
                #
                # value_real, query_real = value_real / value_real.max(), query_real / query_real.max()
                # value_real, query_real = value_real - value_real.mean(), query_real - query_real.mean()
                #
                # value_real[torch.isnan(value_real)] = 1.0
                # value_real = torch.clamp(value_real, min=0.0, max=1.0)
                # # #
                # query_real[torch.isnan(query_real)] = 1.0
                # query_real = torch.clamp(query_real, min=0.0, max=1.0)



                matrix_tmp = (torch.abs(recon_freq - real_freq))# ** 2
                # matrix_tmp = torch.sqrt(matrix_tmp) ** alpha

                # whether to adjust the spectrum weight matrix by logarithm
                # if log_matrix:
                #     matrix_tmp = torch.log(matrix_tmp + 1.0)

                matrix_tmp = matrix_tmp / matrix_tmp.max()

                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()
                #
                # assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                #         'The values of spectrum weight matrix should be in the range [0, 1], '
                #         'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
                #
                # frequency distance using (squared) Euclidean distance
                tmp = (torch.abs(recon_freq - real_freq))# ** 2
                # tmp = pearson(recon_freq, real_freq)


                # tmp = tmp / tmp.max()
                freq_distance = tmp

                # phase_difference = pearson(value_real, query_imag)
                # amplitude_difference = pearson(value_real, query_real)
                # phase_difference = torch.abs(value_imag - query_imag)
                # amplitude_difference = torch.abs(value_real - query_real)

                # freq_distance = torch.exp(-(phase_difference + amplitude_difference)*0.01)
                # freq_distance = torch.abs(phase_difference + amplitude_difference)
                # freq_distance = amplitude_difference #phase_difference# + amplitude_difference
                # freq_distance[torch.isnan(freq_distance)] = 0.0
                # dynamic spectrum weighting (Hadamard product)
                loss = weight_matrix * freq_distance
                loss_bak = loss_bak + loss

        i_bak.append(i)

    return torch.mean(loss_bak)




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
        self.kernel_num = 8

        self.sigmas = nn.Parameter(torch.empty(self.kernel_num, 1), requires_grad=True)
        self.center_shift = nn.Parameter(torch.empty(self.kernel_num, 1),
            requires_grad=True)

        nn.init.uniform_(self.sigmas)
        nn.init.uniform_(self.center_shift)

        self.kernel_size = kernel_size
        self.padding = [self.kernel_size // 2, self.kernel_size // 2]

        self.Norm = nn.BatchNorm2d(moduledim)

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

        query_fft_shift = torch.fft.fft(q, dim=-1).unsqueeze(1)
        # query_real, query_imag = torch.abs(query_fft_shift), torch.angle(query_fft_shift)# / math.pi * 180
        x_bak = torch.fft.fft(x, dim=-1)
        for i in range(self.kernel_num):
            x_shift = self._gaussian_blur(x, i)
            # x_shift = x
            x_shift_bak.append(x_shift)

            value_fft_shift = torch.fft.fft(x_shift, dim=-1)


            # value_real, value_imag = torch.abs(value_fft_shift), torch.angle(value_fft_shift)# / math.pi * 180
            #
            # phase_difference = torch.abs(value_imag - query_imag.unsqueeze(-2))
            # amplitude_difference = torch.abs(value_real - query_real.unsqueeze(-2))
            # weight = torch.exp(-(phase_difference + amplitude_difference))
            #
            #
            # value_filter = query_fft_shift.unsqueeze(1).repeat(1, value_fft_shift.size(1), 1) * (
            #         1 - weight) + x_bak * weight

            # matrix_tmp = (torch.abs(value_fft_shift - query_fft_shift))  # ** 2
            # matrix_tmp = matrix_tmp / matrix_tmp.max()
            # matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            # matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            # weight_matrix = matrix_tmp.clone().detach()
            tmp = pearson(value_fft_shift, query_fft_shift)# ** 2
            freq_distance = tmp
            weight = freq_distance
            value_filter = query_fft_shift.repeat(1, value_fft_shift.size(1), 1) * (
                    1 - weight) + x_bak * weight

            # # value_filter = value_fft_shift# * query_fft_shift.unsqueeze(1).repeat(1, value_fft_shift.size(1), 1)
            #
            fre_xo = torch.abs(torch.fft.ifft(value_filter, dim=-1)) + x
            outputs.append((fre_xo))

        outputs = torch.cat(
            [value_fft.unsqueeze(-2) for value_fft in outputs],
            dim=-2)

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
            loss = torch.exp(-(torch.abs(loss)*0.01))
        else:
            loss = 0.0


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
