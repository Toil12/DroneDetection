"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import math

"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""

import torch
import torch.nn.functional as F
x = torch.tensor([[1,2],[3,4]],dtype=torch.float)
a=F.softmax(x,dim=0)
print(a)
b=F.softmax(x,dim=1)
print(b)

k1=1
k2=3
print(math.e**(k1)/(math.e**(k1)+math.e**(k2)))


