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
    @staticmethod
    def to_gray_image(self,image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def to_equatl_hist(image):
        # Must be gray-scale image input
        return cv2.equalizeHist(image)
    @staticmethod
    def to_laplacian_filter(image):
        return cv2.Laplacian(image, -1, ksize=3)
    @classmethod
    def main_process(clf,image_paths):
        for image_path in image_paths:
            image=cv2.imread(image_path)
            image=cv2.resize(image,(640,640))

            image=clf.to_gray_image(clf,image)
            image=clf.to_equatl_hist(image)
            image=clf.to_laplacian_filter(image)


            # cv2.imshow("original",image)
            # cv2.imshow("gray_scale",img_gray)
            # cv2.imshow("histogram_scale", img_hist)
            cv2.imshow("All",image)
            cv2.waitKey(0)





if __name__ == '__main__':
    test_image="10.jpg"
    processor=ImagePreProcessor.main_process([test_image])