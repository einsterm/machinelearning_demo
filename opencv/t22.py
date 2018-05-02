# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np


# 求出图像均值作为阈值来二值化
def custom_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("before", gray)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w * h])  # 化为一维数组
    mean = m.sum() / (w * h)
    print("mean: ", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("2value", binary)


src = cv.imread("G:/test/559.jpg")
custom_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
