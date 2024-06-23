"""
coding:utf-8
@Time       :2024/5/27 8:33
@Author     :ywLi
@Institute  :DonghaiLab
"""
import argparse

from ultralytics import YOLO
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
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

    # Train the model with all_images GPUs
    results = model.train(data=datapath,
                          epochs=100,
                          imgsz=1280,
                          device=devices,
                          optimizer='Adam')
    results = model(test_data_path)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
    model.export()