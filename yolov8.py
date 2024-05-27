"""
coding:utf-8
@Time       :2024/5/27 8:33
@Author     :ywLi
@Institute  :DonghaiLab
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data=f"yolov5-master/data/usc_5.yaml",
                      epochs=100,
                      imgsz=1280,
                      device=[0,1])