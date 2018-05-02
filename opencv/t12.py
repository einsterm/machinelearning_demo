# -*- coding: utf-8 -*-
import cv2 as cv

src = cv.imread("G:/test/559.jpg")
cv.namedWindow("before", cv.WINDOW_NORMAL)
cv.imshow("before", src)

# 通道分离，输出三个单通道图片
b, g, r = cv.split(src)  # 将彩色图像分割成3个通道
cv.imshow("blue", b)
cv.imshow("green", g)
cv.imshow("red", r)

# 通道合并
src = cv.merge([b, g, r])
cv.imshow("merge", src)

# 修改某个通道的值
src[:, :, 2] = 100
cv.imshow("one_channel", src)

cv.waitKey(0)
cv.destroyAllWindows()
