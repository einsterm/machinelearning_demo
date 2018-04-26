# -*- coding: utf-8 -*-
import cv2
import dlib
import os

output_dir = './myTEST'
size = 190

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
img = cv2.imread("2.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
dets = detector(gray_img, 1)  # 使用detector进行人脸检测
for i, d in enumerate(dets):
    x1 = d.top() if d.top() > 0 else 0
    y1 = d.bottom() if d.bottom() > 0 else 0
    x2 = d.left() if d.left() > 0 else 0
    y2 = d.right() if d.right() > 0 else 0
    face = img[x1:y1, x2:y2]
    face = cv2.resize(face, (size, size))
    cv2.imwrite(output_dir + '/' + str(i) + '.jpg', face)
    print(i)
