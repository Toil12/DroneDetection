"""
coding:utf-8
@Time       :2024/5/27 8:33
@Author     :ywLi
@Institute  :DonghaiLab
"""
from ultralytics import YOLO
import pathlib
import os

datapath=pathlib.Path(os.path.curdir,"usc_5_v8.yaml")

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data=datapath,
                      epochs=100,
                      imgsz=1280,
                      device='cpu',
                      optimizer='Adam')