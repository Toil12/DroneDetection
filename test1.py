"""
coding:utf-8
@Time       :2024/8/4 11:42
@Author     :ywLi
@Institute  :DonghaiLab
"""
import cv2
import matplotlib
import os
from PIL import Image


PATH=f"datasets_processed/ARD-MAV/images/val"

for file in os.listdir(PATH):
    path=os.path.join(PATH,file)
    new_name=file[11:]
    new_path=os.path.join(PATH,new_name)
    os.rename(path,new_path)