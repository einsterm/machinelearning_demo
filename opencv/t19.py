# -*- coding: utf-8 -*-
import cv2 as cv


# 提升对比度（默认提升），只能是灰度图像
def equalHist_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("原来", gray)  # 因为只能处理灰度图像，所以输出原图的灰度图像用于对比
    dst = cv.equalizeHist(gray)
    cv.imshow("默认处理", dst)


# 对比度限制（自定义提示参数）
def clahe_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # clipLimit是对比度的大小，tileGridSize是每次处理块的大小
    dst = clahe.apply(gray)
    cv.imshow("自定义处理", dst)


src = cv.imread("G:/test/559.jpg")
# equalHist_image(src)
clahe_image(src)
