#!/usr/bin/env python

import numpy as np
import skimage
print(skimage.__version__)
import cv2 as cv2
from matplotlib import pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
import os
import sys
sys.path.append("/home/hojat/Desktop/building_detection")
import autoColorDetection as acd


user_contour = [[]]

def resizeImg(img):

    w = 0
    h = 0

    if img.shape[0] > img.shape[1]:
        h = 240
        w = int(240. * img.shape[1] / img.shape[0])
    else:
        w = 320
        h = int(320. * img.shape[0] / img.shape[1])

    return cv2.resize(img, (w,h))

def mouse_click(event, x, y, flags, param):
    global user_contour

    if event == cv2.EVENT_LBUTTONDOWN:
        user_contour[0].append([x,y])

def defineWalls(img):
    global user_contour

    imgCopy = img.copy()

    cv2.namedWindow("define walls")
    cv2.setMouseCallback("define walls", mouse_click)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)*255

    key = ''
    while key != ord('q'):
        if key == ord('d'):
            cv2.drawContours(imgCopy, np.array(user_contour), -1, (0,0,0), thickness=-1)
            cv2.drawContours(mask, np.array(user_contour), -1, 255, thickness=-1)

            user_contour = [[]]

        cv2.imshow("define walls", imgCopy)

        key = cv2.waitKey(10)

    # mask = cv2.bitwise_not(mask)
    cv2.imshow("define walls", mask)
    key = cv2.waitKey()

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return mask

colorDetector = acd.AutoColorDetector()

bld_count = 1
while True:
    sample_file_name = "raw/bld{:d}_0.jpg".format(bld_count)
    if os.path.isfile(sample_file_name):
        sample = cv2.imread(sample_file_name, cv2.COLOR_BGR2RGB)
        sample = resizeImg(sample)

        mask = defineWalls(sample)

        fig, ax = plt.subplots(2)
        labels = colorDetector.detectBuildingColor(sample, mask=mask)
        if labels.any() != None:

            out = color.label2rgb(labels, sample, kind='overlay', bg_label=0)

            ax[0].imshow(out)
            ax[0].set_axis_off()

        bld_count1 = 1
        while True:

            current_file_name = "raw/bld{:d}_{:d}.jpg".format(bld_count, bld_count1)
            if os.path.isfile(current_file_name):
                current = cv2.imread(current_file_name, cv2.COLOR_BGR2RGB)
                current = resizeImg(current)

                labels = colorDetector.detectBuildingColor(current)
                if labels.any() != None:

                    out = color.label2rgb(labels, current, kind='overlay', bg_label=0)

                    ax[1].imshow(out)
                    ax[1].set_axis_off()

                    plt.draw()
                    plt.pause(1)

            else:
                break
            bld_count1 = bld_count1 + 1

    else:
        break

    bld_count = bld_count + 1
