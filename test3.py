"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import torch

import torch

a = torch.arange(15).reshape(5, 3)
r1=a.split((2,1), 1)

print(a)
print(r1)