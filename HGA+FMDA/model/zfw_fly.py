import torch.nn as nn
import torch

class fly(nn.Module):
    """feature fusion"""
    def __init__(self,moduledim = 512):
        super(fly, self).__init__()

        # self.sigmoid=nn.Sigmoid()

    def forward(self,query, value):
        query1=torch.fft.fft(query, dim=1)
        value1 = torch.fft.fft(value, dim=1)
        query1=torch.abs(query1)
        value1=torch.abs(value1)
        V= query1 * value1
        X=nn.Softmax(dim=1)(V)*query+nn.Softmax(dim=1)(V)*value
        # X=torch.fft.ifft(X1,dim=1)
        # X=torch.abs(X)
        batchSize = X.size(0)
        num = X.size(1)
        X = X.contiguous().view(batchSize, -1)
        n = X.shape[0] * X.shape[1]
        d = (X - X.mean(dim=[0, 1])).pow(2)
        s = (X - X.mean(dim=[0, 1]))
        # s1 = (X - X.mean(dim=[0, 1])).pow(3)/6
        v = d.sum(dim=[0, 1]) / (n - 1)
        E_inv = (d + s) / v
        # E_inv=torch.relu(E_inv)
        # E_inv = torch.fft.fft(E_inv, dim=1)
        # E_inv = torch.abs(E_inv)
        xo = nn.Softmax(dim=1)(E_inv) * X
        weight = torch.sigmoid(xo)
        weight = weight.view(batchSize, num, -1)
        xo = query * weight + value * (1 - weight)
        # E_inv=E_inv.view(batchSize, num, -1)
        # xo = nn.Softmax(dim=1)(xo) * query + nn.Softmax(dim=1)(xo) * value
        # xo = V * weight
        return xo
