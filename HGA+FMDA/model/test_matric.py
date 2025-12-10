import numpy as np
import torch

a = np.array((1, 2, 3, 4))
a = torch.tensor(a)
b = a.unsqueeze(0).repeat(4, 1)
c = a.unsqueeze(1).repeat(1, 4)

mask3 = torch.triu(b) + torch.tril(c) - torch.eye(4) * a


print(mask3)

print(a)