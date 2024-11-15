"""
coding:utf-8
@Time       :2024/9/14 9:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import shutil
import platform
import time
import cv2
import os

import xml.dom.minidom as xmldom
import os.path as osp

from sklearn.model_selection import train_test_split
from Image_preprocessor import ImagePreProcessor as imgpp

system=platform.system().lower()
slash=""
if system == 'windows':
    slash="\\"
elif system == 'linux':
    slash="/"

data_video_name="self"
VIDEOS_ROOT = os.path.join(os.getcwd(), "datasets_original", data_video_name)
ANNO_PATH = os.path.join(os.getcwd(), "datasets_original", data_video_name)
IMAGES_ROOT = os.path.join(os.getcwd(), "datasets_local", data_video_name, "video_images")
OUTPUT_IMAGES_DIR = "all_images"
OUTPUT_ANNOTATION_DIR = "all_annotations"
BASE_PATH = os.curdir
def video2imgs(videoPath,imgPath):
    # 目标文件夹不存在，则创建
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)
    # If not empty, cleat directory
    elif len(os.listdir(imgPath))!=0:
        for file_name in os.listdir(imgPath):
            file_path=os.path.join(imgPath,file_name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.islink(file_path):
                shutil.rmtree(file_path)

    video_dir=videoPath.split(".")[0].split(slash)[-1]
    cap = cv2.VideoCapture(videoPath)    # 获取视频
    judge = cap.isOpened()                 # 判断是否能打开成功
    print(judge)
    fps = cap.get(cv2.CAP_PROP_FPS)      # 帧率，视频每秒展示多少张图片
    print('fps:',fps)

    frames = 1                           # 用于统计所有帧数
    count = 1                            # 用于统计保存的图片数量

    while(judge):
        flag, frame = cap.read()         # 读取每一张图片 flag表示是否读取成功，frame是图片
        if not flag:
            print(flag)
            print(f"Process {video_dir} finished!")
            break
        else:
            if frames % 1 == 0:         # 每隔1帧抽一张
                imgname = f'{video_dir}_' + str(count).rjust(4,'0') + ".jpg"
                newPath = os.path.join(imgPath,imgname)
                # print(imgname,newPath)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                count += 1
        frames += 1
    cap.release()
    print("共有 %d 张图片"%(count-1))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

def annotation_xml_to_yolo(images_root,anno_root)->None:
    """
    The images will be put under path/video_images/directory_under_video_name
    :param images_root: The root of the dataset from video transformed images.
    :param anno_root: The root of annotation.
    """

    # delete the buffer directions
    shutil.rmtree(os.path.join(BASE_PATH,OUTPUT_IMAGES_DIR))
    shutil.rmtree(os.path.join(BASE_PATH,OUTPUT_ANNOTATION_DIR))
    os.mkdir(os.path.join(BASE_PATH,OUTPUT_IMAGES_DIR))
    os.mkdir(os.path.join(BASE_PATH,OUTPUT_ANNOTATION_DIR))
    # start re-write
    for video_dir in os.listdir(anno_root):
        for file in os.listdir(os.path.join(anno_root,video_dir)):
            xml_file_path=os.path.join(anno_root,video_dir,file)
            # read xml files
            xml_file = xmldom.parse(xml_file_path)

            # print(f)
            eles = xml_file.documentElement
            # try, if no object, continue
            try:
                xmin = float(eles.getElementsByTagName("xmin")[0].firstChild.data)
                xmax = float(eles.getElementsByTagName("xmax")[0].firstChild.data)
                ymin = float(eles.getElementsByTagName("ymin")[0].firstChild.data)
                ymax = float(eles.getElementsByTagName("ymax")[0].firstChild.data)
            except Exception as e:
                print(f"{file} error")
                continue

            width=float(eles.getElementsByTagName("width")[0].firstChild.data)
            height = float(eles.getElementsByTagName("height")[0].firstChild.data)

            # transform to yolo form
            yolo_x = (xmin +  xmax) / (2*width)
            yolo_y = (ymin + ymax) / (2*height)
            yolo_w = (xmax-xmin) / width
            yolo_h = (ymax-ymin) / height


            with open(f'{BASE_PATH}/{OUTPUT_ANNOTATION_DIR}/{file.split(".")[0]}.txt', mode='w') as f:
                f.write("0" + ' ')
                f.write(str(yolo_x) + ' ')
                f.write(str(yolo_y) + ' ')
                f.write(str(yolo_w) + ' ')
                f.write(str(yolo_h))

            # print(type(images_root),type(video_dir),type(file))
            image_path=os.path.join(images_root,video_dir,file.split(".")[0]+".jpg")

            shutil.copy(image_path,OUTPUT_IMAGES_DIR)


def data_split(agg_pars=None):
    if agg_pars is None:
        agg_pars = {
            "gray": 0,
            "hist": 0,
            "lap": 0
        }
    new_data_images_path=os.path.join(os.curdir,"datasets_processed","ARD-MAV","images")
    new_data_anno_path=os.path.join(os.curdir,"datasets_processed","ARD-MAV","labels")

    output_images_dir=OUTPUT_IMAGES_DIR
    output_anno_dir=OUTPUT_ANNOTATION_DIR



    shutil.rmtree(os.path.join(new_data_images_path,"train"))
    shutil.rmtree(os.path.join(new_data_images_path, "val"))
    shutil.rmtree(os.path.join(new_data_anno_path, "train"))
    shutil.rmtree(os.path.join(new_data_anno_path, "val"))

    os.mkdir(os.path.join(new_data_images_path,"train"))
    os.mkdir(os.path.join(new_data_images_path,"val"))
    os.mkdir(os.path.join(new_data_anno_path, "train"))
    os.mkdir(os.path.join(new_data_anno_path, "val"))

    # Split the data into training and test
    annotations=os.listdir(output_anno_dir)
    images=os.listdir(output_images_dir)
    image_annotation_tuples=list(zip(images,annotations))
    train_tuples, val_tuples = train_test_split(image_annotation_tuples,
                                                 train_size=0.8,
                                                 test_size=0.2,
                                                 shuffle=False
                                                 )

    # Make annotations and images as pairs in training set
    for t in train_tuples:
        with open(osp.join(output_anno_dir, f"{t[1]}")) as f:
            # Drop data which is not with a target in the view, pos in positions < 0
            drop_tag=False
            positions=f.read().split(" ")
            for pos in positions[1:]:
                pos=float(pos)
                if pos<0:
                    drop_tag=True
                    break
        if drop_tag:
            continue
        else:
            img_path=os.path.join(output_images_dir, t[0])

            image=cv2.imread(img_path)

            image=imgpp.main_process(image,agg_pars)
            #
            image_id,file_id=img_path.split(slash)[-1:-3:-1]
            cv2.imwrite(osp.join(new_data_images_path,"train",f"{image_id.split('.')[0]}.jpg"),image)
            shutil.copy(osp.join(output_anno_dir, f"{t[1]}"), osp.join(new_data_anno_path, "train", f"{t[1]}"))

    # Make annotations and images as pairs in validation set
    for t in val_tuples:
        with open(os.path.join(output_anno_dir, f"{t[1]}")) as f:
            # Drop data which is not with a target in the view, pos in positions < 0
            drop_tag = False
            positions = f.read().split(" ")
            for pos in positions[1:]:
                pos = float(pos)
                if pos < 0:
                    drop_tag = True
                    break
        if drop_tag:
            continue
        else:

            img_path =os.path.join(output_images_dir, t[0])
                # print(img_path)
            image = cv2.imread(img_path)
            image_id, file_id = img_path.split(slash)[-1:-3:-1]
            cv2.imwrite(osp.join(new_data_images_path, "val", f"{image_id.split('.')[0]}.jpg"), image)
            shutil.copy(osp.join(output_anno_dir, f"{t[1]}"), osp.join(new_data_anno_path, "val", f"{t[1]}"))




if __name__ == '__main__':

    print("start process")
    start_time=time.time()
    # Videos to frames
    for dir_name in os.listdir(VIDEOS_ROOT):
        video_path=os.path.join(VIDEOS_ROOT,dir_name)
        img_dir_name=dir_name.split(".")[0]
        img_dir_path=os.path.join(IMAGES_ROOT,img_dir_name)

        video2imgs(video_path,img_dir_path)


    # Annotations to yolo form
    # print(IMAGES_ROOT)
    # annotation_xml_to_yolo(IMAGES_ROOT,ANNO_PATH)
    # data_split()

    end_time=time.time()
    print(f"spend {end_time-start_time}s")
