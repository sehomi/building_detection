#!/usr/bin/env python

import numpy as np
import skimage
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

    mask = cv2.bitwise_not(mask)
    mask[0,:] = 0
    mask[-1,:] = 0
    mask[:,0] = 0
    mask[:,-1] = 0
    cv2.imshow("define walls", mask)
    key = cv2.waitKey()

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return mask

def onclick(event):
    global ax, clicked1, clicked2

    if event.inaxes == ax[0]:
        clicked1 = [int(event.xdata), int(event.ydata)]
    elif event.inaxes == ax[1]:
        clicked2 = [int(event.xdata), int(event.ydata)]


def press(event):
    global next, data

    if event.key == 'n':
        next = True
    elif event.key == 's':
        np.savetxt("features/data.csv", data, delimiter=",")

colorDetector = acd.AutoColorDetector()
data = []
clicked1 = []
clicked2 = []
next = False

fig, ax = plt.subplots(2)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', press)

bld_count = 1
while True:
    sample_file_name = "raw/bld{:d}_0.jpg".format(bld_count)
    if os.path.isfile(sample_file_name):
        sample = cv2.imread(sample_file_name, cv2.COLOR_BGR2RGB)
        sample = resizeImg(sample)

        mask = defineWalls(sample)

        labels, masks1, fts1 = colorDetector.detectBuildingColor(sample, mask=mask)
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

                labels, masks2, fts2 = colorDetector.detectBuildingColor(current)
                if labels.any() != None:

                    out = color.label2rgb(labels, current, kind='overlay', bg_label=0)

                    ax[1].imshow(out)
                    ax[1].set_axis_off()

                    plt.draw()

                    pairs = []
                    while not next:

                        clicked1 = []
                        clicked2 = []

                        while len(clicked1) == 0 or len(clicked2) == 0:
                            if next:
                                break
                            plt.pause(0.1)

                        if len(clicked1) == 0 or len(clicked2) == 0:
                            continue

                        d1, i1 = colorDetector.getFts(masks1, fts1, clicked1)
                        d2, i2 = colorDetector.getFts(masks2, fts2, clicked2)

                        if d1 and d2:
                            trueFts = np.hstack((d1, d2))
                            data.append( np.hstack( (trueFts,np.array([1])) ))
                            pairs.append([i1,i2])


                    next = False

                    for i in range(len(masks1)):
                        for j in range(len(masks2)):

                            if [i,j] in pairs or [j,i] in pairs:
                                continue

                            falseFts = np.hstack((fts1[i], fts2[j]))
                            data.append( np.hstack( (falseFts,np.array([0])) ))
                            pairs.append([i,j])

                    print(data)
            else:
                break
            bld_count1 = bld_count1 + 1

    else:
        break

    bld_count = bld_count + 1
