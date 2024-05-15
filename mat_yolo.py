"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import scipy
import numpy as np
import os

from PIL import Image

# The keys of matlab file is as the followed, box is the item including data:
# __header__
# __version__
# __globals__
# box
annotations = scipy.io.loadmat(f"datasets/usc_5/annotations/anotation.mat")

# Get the file name for the label file process
images_training=[x.split('.')[0] for x in os.listdir(f"datasets/usc_5/images/train")]
image_example=Image.open(f'datasets/usc_5/images/train/{os.listdir(f"datasets/usc_5/images/train")[0]}')
image_example=np.array(image_example)
h,w,c=image_example.shape

# Each data is with 4 features, x_center, y_center, width, height in pixels
labels = annotations['box']
labels = np.array(labels)
# print(labels[0:5])
# print(type(labels[0][0]))

labels_training_path=f"datasets/usc_5/labels/train"

for i in range(len(labels)):
    file_name=str((i+1)*10)
    with open(f'{labels_training_path}/{file_name}.txt', mode='w') as f:
        f.write("0"+' ')
        f.write(str(labels[i][0] / w) + ' ')
        f.write(str(labels[i][1] / h) + ' ')
        f.write(str(labels[i][2] / w) + ' ')
        f.write(str(labels[i][3] / h) + ' ')




