# -*- coding: utf-8 -*-
import cv2 as cv


# 数值运算：加减乘除
def shu_image(src11, src22):
    src = cv.add(src11, src22)  # 加
    cv.imshow("add", src)
    src = cv.subtract(src11, src22)  # 减
    cv.imshow("reduce", src)
    src = cv.divide(src11, src22)  # 乘
    cv.imshow("ride", src)
    src = cv.multiply(src11, src22)  # 除
    cv.imshow("divide", src)


src1 = cv.imread("G:/test/559.jpg")
src2 = cv.imread("G:/test/0.jpg")
cv.imshow("before1", src1)
cv.imshow("before2", src2)
shu_image(src1, src2)
cv.waitKey(0)
cv.destroyAllWindows()
