"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""

import scipy

annots=scipy.io.loadmat(f"dataset_local/5/anotation.mat")
# print(annots)
for key,value in annots.items():
    print(key,value)