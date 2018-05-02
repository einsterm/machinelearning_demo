# -*- coding: utf-8 -*-
import cv2 as cv


# numpy数组操作
def access_pixles(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    print("width : %s, height : %s, channel : %s" % (width, height, channel))
    for row in range(height):
        for col in range(width):
            for c in range(channel):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("after", image) #处理后的图像


src = cv.imread("G:/test/559.jpg")
cv.imshow("before", src)
t1 = cv.getTickCount()  # 毫秒级别的计时函数,记录了系统启动以来的时间毫秒
access_pixles(src)
t2 = cv.getTickCount()
time = (t2 - t1) * 1000 / cv.getTickFrequency()  # getTickFrequency用于返回CPU的频率,就是每秒的计时周期数
print("time: %s" % time)  # 输出运行的时间
cv.waitKey(0)
cv.destroyAllWindows()
