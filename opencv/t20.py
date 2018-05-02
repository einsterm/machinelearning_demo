# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np


# 模版匹配
def template_image():
    tpl = cv.imread("G:/test/6.jpg")
    target = cv.imread("G:/test/559.jpg")
    cv.imshow("temple", tpl)
    cv.imshow("src", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("mach" + np.str(md), target)


template_image()
cv.waitKey(0)
cv.destroyAllWindows()
