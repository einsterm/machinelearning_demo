# -*- coding: utf-8 -*-
import cv2 as cv


# 高斯金字塔
def pyramid_image(image):
    level = 3  # 金字塔的层数
    temp = image.copy()  # 拷贝图像
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("guss" + str(i), dst)
        temp = dst.copy()
    return pyramid_images


# 拉普拉斯金字塔
def laplian_image(image):
    pyramid_images = pyramid_image(image)
    level = len(pyramid_images)
    for i in range(level - 1, -1, -1):
        if (i - 1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("laplian" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            cv.imshow("laplian" + str(i), lpls)


src = cv.imread("G:/test/8.jpg")
cv.imshow("before", src)
laplian_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
