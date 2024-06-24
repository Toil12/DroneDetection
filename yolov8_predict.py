"""
coding:utf-8
@Time       :2024/6/24 11:12
@Author     :ywLi
@Institute  :DonghaiLab
"""
import torch

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Create a random torch tensor of BCHW shape (1, 3, 640, 640) with values in range [0, 1] and type float32
source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

# Run inference on the source
results = model(source)  # list of Results objects