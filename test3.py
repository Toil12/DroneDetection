"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import scipy
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from PIL import Image

# The keys of matlab file is as the followed, box is the item including data:
# __header__
# __version__
# __globals__
# box
with open("datasets/coco8/labels/train/000000000025.txt",mode="r") as f:
    for line in f.readlines():
        image=cv2.imread(f'datasets/coco8/images/train/000000000025.jpg')
        h,w,c=image.shape
        # print(line.split()[1]*w)
        center_x=int(float(line.split()[1])*w)
        center_y = int(float(line.split()[2])*h)
        center_w = int(float(line.split()[3])*w)
        center_h = int(float(line.split()[4])*h)

        #
        # print((org_x-org_w/2, org_y-org_h/2), (org_x+org_w/2, org_y+org_h/2))
        cv2.rectangle(image, (int(center_x-center_w/2), int(center_y-center_h/2)),
                      (int(center_x+center_w/2), int(center_y+center_h/2)), (0, 0, 255), 2)

        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




