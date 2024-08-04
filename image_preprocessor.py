"""
coding:utf-8
@Time       :2024/8/4 11:42
@Author     :ywLi
@Institute  :DonghaiLab
"""
import cv2
import matplotlib

from PIL import Image
class ImagePreProcessor:
    def __init__(self,image_paths):
        self.image_paths=image_paths
        self.main_process()

    def to_gray_image(self,image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    def to_equatl_hist(self,image):
        # Must be gray-scale image input
        return cv2.equalizeHist(image)

    def to_laplacian_filter(self,image):
        return cv2.Laplacian(image, -1, ksize=3)
    def main_process(self):
        for image_path in self.image_paths:
            image=cv2.imread(image_path)
            image=cv2.resize(image,(640,640))

            image=self.to_gray_image(image)
            image=self.to_equatl_hist(image)
            image=self.to_laplacian_filter(image)


            # cv2.imshow("original",image)
            # cv2.imshow("gray_scale",img_gray)
            # cv2.imshow("histogram_scale", img_hist)
            cv2.imshow("All",image)
            cv2.waitKey(0)





if __name__ == '__main__':
    test_image="10.jpg"

    processor=ImagePreProcessor([test_image])