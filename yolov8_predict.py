"""
coding:utf-8
@Time       :2024/6/24 11:12
@Author     :ywLi
@Institute  :DonghaiLab
"""
import os.path
import time

import matplotlib.pyplot as plt
import cv2
import os

from ultralytics import YOLO

ROOT = os.getcwd()
# Load a pretrained YOLOv8n model
model = YOLO(f"./trained_models/best.pt")

# Get the input image and the corresponding
file_name = "0101280"

img_source= rf"D:\Codes\python\DroneDetection\{file_name}.jpg"
video_source = os.path.join(ROOT,"phantom47.mp4")
anno_source = rf"D:\Codes\python\DroneDetection\datasets\usc_all\labels\train\{file_name}.txt"

# Plot the original image
# fix, ax = plt.subplots(2, 1)
# img_bgr = cv2.imread(img_source)
# height, width, channel = img_bgr.shape
# print(width, height)
# ax[0].imshow(img_bgr[:, :, ::-1])
# ax[0].set_title("Original Image")

# Run inference on the source
start_time=time.time()
results = model(video_source, save=True, show=True)  # list of Results objects
end_time=time.time()
print(f"use {end_time-start_time}s for inference")

# The following is the part for customizing the
# print(len(results))
# results[0].show()

# # Get predicted boxes keypoints
# bboxes_xyxy=results[0].boxes.xyxy.cpu().numpy().astype('uint32')
# bboxes_keypoints=results[0].names
# confidence=results[0].boxes.conf
# num_box=len(results[0].boxes.cls)
# print(bboxes_xyxy,bboxes_keypoints,confidence,num_box)
#
# # Plot boxes as predicted
# img_bgr_predict=img_bgr.copy()
# for i in range(num_box):
#     img_bgr_predict = cv2.rectangle(img_bgr,
#                             (bboxes_xyxy[i][0], bboxes_xyxy[i][1]),
#                             (bboxes_xyxy[i][2], bboxes_xyxy[i][3]),
#                             (0,0,150),
#                             6)
#
# # Read label boxes, x y w h
# img_bgr_label=img_bgr_predict
# with open(anno_source) as f:
#     for line in f.readlines():
#         c,x,y,w,h= [float(x) for x in line.split(" ")]
#         img_bgr_label = cv2.rectangle(img_bgr_predict,
#                                 (int((x-w/2)*width), int((y+h/2)*height)),
#                                 (int((x+w/2)*width), int((y-h/2)*height)),
#                                 (0, 150, 0),
#                                 6)
# ax[1].imshow(img_bgr_label[:,:,::-1])
# ax[1].set_title("Red prediction/Green label")
#
# plt.tight_layout()
# plt.show()
# plt.savefig("prediction_result.png")
