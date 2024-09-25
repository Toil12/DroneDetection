"""
coding:utf-8
@Time       :2024/5/27 8:33
@Author     :ywLi
@Institute  :DonghaiLab
"""
from ultralytics import YOLO,RTDETR
import argparse
import pathlib
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
DATA_ROOT=os.path.join(os.getcwd(),"datasets_processed")
TRAINED_MODEL_ROOT=os.path.join(os.getcwd(),"trained_models")
YOLO_DATA_ROOT=os.path.join(os.getcwd(),"yolo_data_files")
# print(TRAINED_MODEL_ROOT,YOLO_DATA_ROOT,DATA_ROOT)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",default="yolo_v8",type=str)
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--imgsz",default=640,type=int)
    parser.add_argument("--data_path",default="usc_5_v8.yaml",type=str)

    args=parser.parse_args()


    datapath=pathlib.Path(YOLO_DATA_ROOT,args.data_path)
    # print(datapath)
    # testpath=pathlib.Path(os.path.curdir,"")

    devices_count=torch.cuda.device_count()
    devices=[x for x in range(devices_count)]
    # print(devices)

    # Load a model
    model=0
    if args.model_name=="yolo_v8":
        model_path=os.path.join(TRAINED_MODEL_ROOT,"yolov8s.pt")  # load a pretrained model (recommended for training)
        print(model_path)
        model=YOLO(model_path)
    elif args.model_name=="yolo_v10":
        model_path=os.path.join(TRAINED_MODEL_ROOT,"yolov10s.pt")
        model=YOLO(model_path)
    elif args.model_name=="rtdetr":
        model_path = os.path.join(TRAINED_MODEL_ROOT, "rtdetr-x.pt")
        model=RTDETR(model_path)
    # Train the model with all_images GPUs
    results = model.train(data=datapath,
                          epochs=args.epochs,
                          imgsz=args.imgsz,
                          device="cuda:1",
                          optimizer='Adam')