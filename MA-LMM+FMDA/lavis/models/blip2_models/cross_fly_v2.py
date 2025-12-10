import torch.nn as nn
import torch



class pearson(torch.nn.Module):
    def __init__(self, moduledim=512, channels=None, e_lambda=1e-4):
        super(pearson, self).__init__()

        self.activaton = nn.LeakyReLU()
        self.e_lambda = e_lambda
        # self.Norm = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.Norm2 = nn.BatchNorm1d(moduledim, eps=1e-05)
        # self.q_porj = nn.Linear(512, 512)

    def pearson_distance(self, x, y):
        mean_x = x.mean(dim=[-2], keepdim=True)
        mean_y = y.mean(dim=[-2], keepdim=True)
        c = torch.tensor([True]).cuda()
        d = torch.tensor([0]).cuda()
        # 计算 Pearson 相关系数的分子部分
        cov_xy = (x - mean_x) * (y - mean_y)

        # 计算 Pearson 相关系数的分母部分
        # a = torch.std(x, unbiased=True)
        a = (x - mean_x).pow(2).sum(dim=[-2], keepdim=True)

        # b = torch.std(y, unbiased=True)
        b = (y - mean_y).pow(2).sum(dim=[-2], keepdim=True)
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

        weight = (torch.abs(x - y)) ** 2
        weight = torch.exp(-(weight) * 0.5)
        # print(weight.max(), weight.min())
        weight = weight * weight_matrix


        # print(weight.max(), weight.min(), weight.mean())
        return weight

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



class channel_crossmodel(torch.nn.Module):
    def __init__(self, moduledim, kernel_size, dsl_init_sigma, dsl_init_center, channels=None, e_lambda=1e-4):
        super(channel_crossmodel, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        self.Norm = nn.BatchNorm1d(moduledim)
        self.pearson = pearson_channel()

        self.kernel_size = kernel_size

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
        q_bak = q


        query_fft_shift = torch.fft.fft(q, dim=-2)  # 不能随意修改形状，会造成梯度混乱。

        value_fft_shift = torch.fft.fft(x, n=q.size(-2), dim=-2)


        weight = self.pearson(value_fft_shift, query_fft_shift)  # ** 2

        # value_filter = query_fft_shift * (1 - weight) + value_fft_shift * weight
        value_filter = value_fft_shift * weight

        value_outputs = torch.abs(torch.fft.ifft(value_filter, n=x.size(-2), dim=-2))

        # query_filter = query_fft_shift * (1-weight)
        #
        # query_outputs = torch.abs(torch.fft.ifft(query_filter, dim=-2))
        #
        # outputs = torch.cat((value_outputs,query_outputs), dim=-2)

        return value_outputs


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
        value = self.cross(value, query)
        return value
