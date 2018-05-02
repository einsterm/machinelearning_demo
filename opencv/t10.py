# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np


def create_image_white_1():
    img = np.zeros([400, 400, 3], np.uint8)  # zeros:double类零矩阵  创建400*400 3个通道的矩阵图像 参数时classname为uint8
    img[:, :, 0] = np.ones([400, 400]) * 255  # ones([400, 400])是创建一个400*400的全1矩阵，*255即是全255矩阵 并将这个矩阵的值赋给img的第一维
    img[:, :, 1] = np.ones([400, 400]) * 255  # 第二维全是255
    img[:, :, 2] = np.ones([400, 400]) * 255  # 第三维全是255
    cv.imshow("create_image_white_1", img)  # 输出一张400*400的白色图片(255 255 255):蓝(B)、绿(G)、红(R)


def create_image_white_2():
    img = np.ones([400, 400, 3], np.uint8)
    img[:, :, 0] = img[:, :, 0] * 255
    img[:, :, 1] = img[:, :, 1] * 255
    img[:, :, 2] = img[:, :, 2] * 255
    cv.imshow("create_image_white_2", img)


# 自定义一张单通道的图片
def create_image_one_channel():
    img = np.ones([400, 400, 1], np.uint8)
    img = img * 60
    cv.imshow("create_image_one_channel", img)


# 像素取反
def image_inverse(image):
    dst = cv.bitwise_not(image)
    cv.imshow("image_inverse", dst)


# 色彩空间的转换
def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # RGB转换为gray
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # RGB转换为hsv  H：0-180  S: 0-255 V： 0-255
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)  # RGB转换为yuv
    cv.imshow("yuv", yuv)


# 指定颜色替换
def fill_image(image):
    copyImage = image.copy()  # 复制原图像
    h, w = image.shape[:2]  # 读取图像的宽和高
    mask = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
    cv.floodFill(copyImage, mask, (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill", copyImage)


# 指定位置填充
def fill2_image():
    image = np.zeros([200, 200, 3], np.uint8)
    cv.imshow("before", image)
    mask = np.ones([202, 202, 1], np.uint8)
    mask[100:150, 100:150] = 0
    cv.floodFill(image, mask, (100, 100), (0, 0, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill", image)


def mo_image(src1):
    src2 = cv.blur(src1, (5, 5))
    cv.imshow("blur", src2)  # 均值模糊

    src2 = cv.medianBlur(src1, 5)
    cv.imshow("medianBlur", src2)  # 中值模糊

    src2 = cv.GaussianBlur(src1, (5, 5), 2)
    cv.imshow("GaussianBlur", src2)  # 高斯模糊

    src2 = cv.bilateralFilter(src1, 5, 5, 2)
    cv.imshow("bilateralFilter", src2)  # 双边滤波


# 自定义模糊函数
def zi_image(src1):
    kernel1 = np.ones((5, 5), np.float) / 25  # 自定义矩阵，并防止数值溢出
    src2 = cv.filter2D(src1, -1, kernel1)
    cv.imshow("after1", src2)  # 自定义均值模糊
    kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    src2 = cv.filter2D(src1, -1, kernel2)
    cv.imshow("after2", src2)  # 自定义锐化


if __name__ == '__main__':
    image = cv.imread("G:/test/559.jpg")
    cv.imshow("before", image)
    # create_image_white_1()
    # create_image_white_2()
    # create_image_one_channel()
    # image_inverse(image)
    # color_space_demo(image)
    # fill_image(image)
    # fill2_image()
    # mo_image(image)
    zi_image(image)
    cv.waitKey(0)
    cv.destroyAllWindows()
