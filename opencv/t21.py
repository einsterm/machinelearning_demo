# -*- coding: utf-8 -*-
import cv2 as cv


# 图像二值化 0白色 1黑色  
# 全局阈值  
def threshold_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("原来", gray)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    print("阈值：%s" % ret)
    cv.imshow("total1", binary)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)  # TRIANGLE法,，全局自适应阈值, 参数0可改为任意数字但不起作用，适用于单个波峰
    print("阈值：%s" % ret)
    cv.imshow("total2", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)  # 自定义阈值为150,大于150的是白色 小于的是黑色
    print("阈值：%s" % ret)
    cv.imshow("150wb", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)  # 自定义阈值为150,大于150的是黑色 小于的是白色
    print("阈值：%s" % ret)
    cv.imshow("150bw", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TRUNC)  # 截断 大于150的是改为150  小于150的保留
    print("阈值：%s" % ret)
    cv.imshow("150break", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TOZERO)  # 截断 小于150的是改为150  大于150的保留
    print("阈值：%s" % ret)
    cv.imshow("150remain", binary)

    binary1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("part_mean", binary1)
    binary2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)  # 高斯处理
    cv.imshow("part_gaussian", binary2)


src = cv.imread("G:/test/559.jpg")
threshold_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
