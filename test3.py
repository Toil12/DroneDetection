"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import torch

# Model loading
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

# Inference on images
img = "https://ultralytics.com/images/zidane.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images

# Run inference
results = model(img)

# Display results
results.print()  # Other options: .show(), .save(), .crop(), .pandas(), etc.