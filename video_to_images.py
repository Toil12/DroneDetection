"""
coding:utf-8
@Time       :2024/9/14 9:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import shutil

import cv2
import os
VIDEOS_PATH=os.path.join(os.getcwd(),"dataset_local","ARD-MAV","videos")
ANNO_PATH=os.path.join(os.getcwd(),"dataset_local","ARD-MAV","Annotations")
IMAGES_PATH=os.path.join(os.getcwd(),"datasets_processed","video_images")

def video2imgs(videoPath,imgPath):
    # 目标文件夹不存在，则创建
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)
    # If not empty, cleat directory
    elif len(os.listdir(imgPath))!=0:
        for file_name in os.listdir(videoPath):
            file_path=os.path.join(videoPath,file_name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.islink(file_path):
                shutil.rmtree(file_path)

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
            print("Process finished!")
            break
        else:
            if frames % 10 == 0:         # 每隔10帧抽一张
                imgname = 'jpgs_' + str(count).rjust(3,'0') + ".jpg"
                newPath = imgPath + imgname
                print(imgname)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                count += 1
        frames += 1
    cap.release()
    print("共有 %d 张图片"%(count-1))

if __name__ == '__main__':
    for dir_name in os.listdir(VIDEOS_PATH):
        video_path=os.path.join(VIDEOS_PATH,dir_name.split(".")[0])


    # video2imgs(VIDEOS_PATH,IMAGES_PATH)
