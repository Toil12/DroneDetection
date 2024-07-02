"""
coding:utf-8
@Time       :2024/5/27 8:33
@Author     :ywLi
@Institute  :DonghaiLab
"""
from ultralytics import YOLO
import argparse
import pathlib
import os
import torch

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    datapath=pathlib.Path(os.path.curdir,"usc_5_v8.yaml")
    testpath=pathlib.Path(os.path.curdir,"")

    devices_count=torch.cuda.device_count()
    devices=[x for x in range(devices_count)]
    # print(devices)

    # Load a model
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

    # Train the model with all_images GPUs
    results = model.train(data=datapath,
                          epochs=100,
                          imgsz=1280,
                          device=devices,
                          optimizer='Adam')