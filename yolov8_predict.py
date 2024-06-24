"""
coding:utf-8
@Time       :2024/6/24 11:12
@Author     :ywLi
@Institute  :DonghaiLab
"""
import matplotlib.pyplot as plt
import torch
import cv2

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(f"./models/v8_17.pt")

# Create a random torch tensor of BCHW shape (1, 3, 640, 640) with values in range [0, 1] and type float32
source = r"D:\Codes\python\DroneDetection\datasets\usc_all\images\train\1_20.jpg"

# Run inference on the source
results = model(source)  # list of Results objects
# get box points
bboxes_xyxy=results[0].boxes.xyxy.cpu().numpy().astype('uint32')
bboxes_keypoints=results[0].names
confidence=results[0].boxes.conf
num_box=len(results[0].boxes.cls)
print(bboxes_xyxy,bboxes_keypoints,confidence,num_box)

img_bgr=cv2.imread(source)
plt.imshow(img_bgr[:,:,::-1])
plt.show()
# plot boxes
for i in range(num_box):
    img_bgr = cv2.rectangle(img_bgr,
                            (bboxes_xyxy[i][0], bboxes_xyxy[i][1]),
                            (bboxes_xyxy[i][2], bboxes_xyxy[i][3]),
                            (150,0,0),
                            6)
plt.imshow(img_bgr[:,:,::-1])
plt.show()