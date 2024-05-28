"""
coding:utf-8
@Time       :2024/5/13 16:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import scipy
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from PIL import Image

# The keys of matlab file is as the followed, box is the item including data:
# __header__
# __version__
# __globals__
# box

annotations = scipy.io.loadmat(f"datasets/usc_5/annotations/anotation.mat")
labels_training_path=f"datasets/usc_5/labels/train"
labels_val_path=f"datasets/usc_5/labels/val"
image_path=f"datasets/usc_5/images/train"

# Get the file name for the label file process

images_training=[x.split('.')[0] for x in os.listdir(image_path)]

image_example=Image.open(f'{image_path}/{os.listdir(f"datasets/usc_5/images/train")[0]}')
image_example=np.array(image_example)
w,h,c=image_example.shape
# print(w,h,c)

# Each data is with 4 features, x_center, y_center, width, height in pixels
labels = annotations['box']
labels = np.array(labels)
# print(labels[0:5])
# print(type(labels[0][0]))



# Transform the original coordination to the yolo form image by image
for i in range(len(labels)):
    org_x=labels[i][0]
    org_y=labels[i][1]
    org_w=labels[i][2]
    org_h=labels[i][3]

    yolo_x= (org_x+org_w/2)/ w
    yolo_y= (org_y+org_h)/ h
    yolo_w= org_w/ w
    yolo_h= org_h/ h

    file_name = str((i + 1) * 10)
    # Visualize the annotation
    # image=cv2.imread(f'{image_path}/{file_name}.jpg')
    # print((org_x-org_w/2, org_y-org_h/2), (org_x+org_w/2, org_y+org_h/2))
    # cv2.rectangle(image, (int(org_x), int(org_y)),
    #               (int(org_x+org_w), int(org_y+org_h)), (0, 0, 255), 2)
    #
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Regenerate the annotation data
    with open(f'{labels_training_path}/{file_name}.txt', mode='w') as f:
        f.write("0"+' ')
        f.write(str(yolo_x) + ' ')
        f.write(str(yolo_y) + ' ')
        f.write(str(yolo_w) + ' ')
        f.write(str(yolo_h) + ' ')

    with open(f'{labels_val_path}/{file_name}.txt', mode='w') as f:
        f.write("0"+' ')
        f.write(str(yolo_x) + ' ')
        f.write(str(yolo_y) + ' ')
        f.write(str(yolo_w) + ' ')
        f.write(str(yolo_h) + ' ')




