# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np


# 截取图片中的指定区域或在指定区域添加某一图片
def jie_image(src1):
    src2 = src1[5:89, 200:330]  # 截取第5行到89行的第500列到630列的区域
    cv.imshow("cut", src2)
    src1[105:189, 300:430] = src2  # 指定位置填充，大小要一样才能填充
    cv.imshow("merge", src1)


src = cv.imread("G:/test/559.jpg")
cv.imshow("before", src)
jie_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
