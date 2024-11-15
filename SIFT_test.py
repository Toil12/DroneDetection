"""
coding:utf-8
@Time       :2024/11/15 9:23
@Author     :ywLi
@Institute  :DonghaiLab
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = f"datasets_local/self/video_images/MVI_0519/MVI_0519_0001.jpg"
imgname2 = f"datasets_local/self/video_images/MVI_0519/MVI_0519_1000.jpg"

img1=cv2.imread(imgname1,0)
img2=cv2.imread(imgname2,0)
#实例化的sift函数
sift=cv2.SIFT_create()

#
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)


#
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 使用KNN匹配器进行匹配
matches = flann.knnMatch(des1, des2, k=2)

# 选择优秀的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示匹配结果
# cv2.imshow('Matches', img_matches)
cv2.imwrite("match.jpg",img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()


