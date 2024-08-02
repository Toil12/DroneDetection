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
    parser.add_argument("--model_name",default="yolo_v8",type=str)
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--imgsz",default=640,type=int)
    parser.add_argument("--data_path",default="usc_5_v8.yaml",type=str)

    args=parser.parse_args()


    datapath=pathlib.Path(os.path.curdir,args.data_path)
    # testpath=pathlib.Path(os.path.curdir,"")

    devices_count=torch.cuda.device_count()
    devices=[x for x in range(devices_count)]
    # print(devices)

    # Load a model
    model=0
    if args.model_name=="yolo_v8":
        model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
    elif args.model_name=="yolo_v10":
        model = YOLO("yolov10s.pt")
    # Train the model with all_images GPUs
    results = model.train(data=datapath,
                          epochs=args.epochs,
                          imgsz=args.imgsz,
                          device=devices,
                          optimizer='Adam')