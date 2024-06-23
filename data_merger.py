import scipy
import numpy as np
import os
import os.path as osp
import shutil
import time

from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split

IN_DIRECT="data_complete"
OUTPUT_IMAGES_DIRECT="all_images"
OUTPUT_ANNOTATION_DIRECT="all_annotations"
BASE_PATH=osp.join(os.curdir)
# print(BASE_PATH)
# print(os.path.exists(BASE_PATH))

class DataMerger():
    def __init__(self,in_direct=IN_DIRECT,out_direct=(OUTPUT_IMAGES_DIRECT,OUTPUT_ANNOTATION_DIRECT),base_path=BASE_PATH):
        self.source=osp.join(base_path,in_direct)
        self.output_image=osp.join(base_path,out_direct[0])
        self.output_anno = osp.join(base_path, out_direct[1])
        self.source_directs=os.listdir(self.source)
    def data_merger(self):
        shutil.rmtree(self.output_image)
        shutil.rmtree(self.output_anno)
        os.mkdir(self.output_image)
        os.mkdir(self.output_anno)
        # redirect the
        for direct in self.source_directs:
            # print(self.source)
            images_paths=glob(osp.join(self.source,direct,"*.jpg"))
            annotation_path=glob(osp.join(self.source,direct,"*.mat"))[0]
            self.re_annotate(annotation_path,images_paths,direct)
    def re_annotate(self,label_path,image_paths,direct_id):
        image_example = Image.open(f'{image_paths[0]}')
        image_example = np.array(image_example)
        h, w, c = image_example.shape
        annotations=scipy.io.loadmat(label_path)
        labels = annotations['box']
        labels = np.array(labels)

        for i in range(len(labels)):
            org_x = labels[i][0]
            org_y = labels[i][1]
            org_w = labels[i][2]
            org_h = labels[i][3]

            yolo_x = (org_x + org_w / 2) / w
            yolo_y = (org_y + org_h / 2) / h
            yolo_w = org_w / w
            yolo_h = org_h / h
            # print(yolo_x,yolo_y,yolo_w,yolo_h)
            file_name = f"{direct_id}_{((i + 1) * 10)}"

            # store annotations
            with open(f'{self.output_anno}/{file_name}.txt', mode='w') as f:
                f.write("0" + ' ')
                f.write(str(yolo_x) + ' ')
                f.write(str(yolo_y) + ' ')
                f.write(str(yolo_w) + ' ')
                f.write(str(yolo_h) + ' ')

            # store name-modified images
            for path in image_paths:
                with open(f'{self.output_image}/{file_name}.txt', mode='w') as f:
                    f.write(path)
    def data_split(self):
        annotations=os.listdir(self.output_anno)
        images=os.listdir(self.output_image)
        image_annotation_tuples=list(zip(images,annotations))
        train_tuples, test_tuples = train_test_split(image_annotation_tuples,
                                                     train_size=0.9,
                                                     test_size=0.1,
                                                     shuffle=True
                                                     )
        print(len(train_tuples))
        print(len(test_tuples))




        # print(annotations)
        # print(images)



if __name__ == '__main__':
    s_time=time.time()
    dm=DataMerger()
    # dm.data_merger()
    dm.data_split()
    e_time=time.time()
    print(f"cost {e_time-s_time}")