# -*- coding: utf-8 -*-
import cv2 as cv


# 逻辑运算：与或非的操作
def luo_image(src11, src22):
    src = cv.bitwise_and(src11, src22)  # 与 两张图片同一位置的色素两个值均不为零的才会有输出
    cv.imshow("and", src)
    src = cv.bitwise_or(src11, src22)  # 或 两张图片同一位置的色素两个值不全为零的才会有输出
    cv.imshow("or", src)
    src = cv.bitwise_not(src11)  # 非 对一张图片操作  取反
    cv.imshow("not", src)
    src = cv.bitwise_xor(src11, src22)  # 异或 两张图片同一位置的色素两个值有一个为零，另一个不为零才会输出
    cv.imshow("or_not", src)


src1 = cv.imread("G:/test/559.jpg")
src2 = cv.imread("G:/test/0.jpg")
cv.imshow("before1", src1)
cv.imshow("before2", src2)
luo_image(src1, src2)
cv.waitKey(0)
cv.destroyAllWindows()
