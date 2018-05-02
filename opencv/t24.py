# -*- coding: utf-8 -*-
import cv2 as cv


# 图像梯度：索贝尔算子
def sobel_image(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)  # x方向导数
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)  # y方向导数
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("X", gradx)  # 颜色变化在水平分层
    cv.imshow("Y", grady)  # 颜色变化在垂直分层
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("XY", gradxy)


# 图像梯度：scharr算子：增强边缘
def scharr_image(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)  # x方向导数
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)  # y方向导数
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("X", gradx)  # 颜色变化在水平分层
    cv.imshow("Y", grady)  # 颜色变化在垂直分层
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("XY", gradxy)


# 拉普拉斯算子
def lapalian_image(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("lpls", lpls)


src = cv.imread("G:/test/559.jpg")
cv.imshow("before", src)
# sobel_image(src)
# scharr_image(src)
lapalian_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
