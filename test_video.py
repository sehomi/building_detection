#!/usr/bin/env python

import numpy as np
import skimage
import cv2 as cv2
from matplotlib import pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
import os
import sys
from tensorflow import keras
import pickle
import autoColorDetection as acd


def resizeImg(img, hd, wd):

    w = 0
    h = 0

    if img.shape[0] > img.shape[1]:
        h = hd
        w = int(float(hd) * img.shape[1] / img.shape[0])
    else:
        w = wd
        h = int(float(wd) * img.shape[0] / img.shape[1])

    return cv2.resize(img, (w,h))

def predict(x):
    global model, scaler

    X = np.array([x])
    X = scaler.transform(X)

    res = model.predict(X)

    return res[0,0] > 0.3

model = keras.models.load_model('models/model2')
scaler = pickle.load(open('models/scaler2.pkl','rb'))
colorDetector = acd.AutoColorDetector()

sample = cv2.imread("dataset/association/raw/bld1_0.jpg", cv2.COLOR_BGR2RGB)
sample = resizeImg(sample, 600, 800)

labels1, masks1, fts1 = colorDetector.detectBuildingColor(sample)

cap = cv2.VideoCapture('video/video4.avi')
key = ''

while(cap.isOpened()):

    ret, current = cap.read()

    if ret != True:
        print("Error reading video file.")
        break

    current = resizeImg(current, 600, 800)
    copy = current.copy()

    if key == ord('s'):
        labels2, masks2, fts2 = colorDetector.detectBuildingColor(current)

        for i in range(len(masks1)):
            for j in range(len(masks2)):

                if i == j:
                    continue

                testFts = np.hstack((fts1[i], fts2[j]))

                if predict(testFts):

                    red = np.zeros((masks2[j].shape[0], masks2[j].shape[1], 3), dtype=np.uint8)
                    red[masks2[j] == 255] = (0,0,255)
                    copy = cv2.addWeighted(copy, 1, red, 0.4, 0.0)

        cv2.imshow("sample", sample)
        cv2.imshow("copy", copy)
        cv2.waitKey()

    cv2.imshow("sample", sample)
    cv2.imshow("copy", copy)
    key = cv2.waitKey(33)
