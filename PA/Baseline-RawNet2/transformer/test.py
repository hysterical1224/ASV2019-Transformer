import torch
import torch.nn as nn
inputs = torch.randn((2,10,512,64))
# print(inputs.shape)
odim = 120
conv1 = nn.Conv2d(10, odim, 3, 2)
x = conv1(inputs)
print("x:",x.shape)
mask = 2
print(mask)
mask = mask[:, :, :-2:2]
print(mask.shape)
inputs.masked_fill_(~mask.unsqueeze(-1), 0)


inputs = inputs.unsqueeze(1)



import torch
a=torch.tensor([[[5,5,5,5], [6,6,6,6], [7,7,7,7]], [[1,1,1,1],[2,2,2,2],[3,3,3,3]]])
print(a)
print(a.size())
print("#############################################3")
mask = torch.ByteTensor([[[1],[1],[0]],[[0],[1],[1]]])
print(mask.size())
b = a.masked_fill(mask, value=torch.tensor(-10))
print(b)
print(b.size())






