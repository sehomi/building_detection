#!/usr/bin/env python

import numpy as np
import skimage
import cv2 as cv2
from matplotlib import pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
import os
import sys
import argparse
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

    return res[0,0] > 0.9

def showHSV(img, num):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    cv2.imshow("Split H {:d}".format(num),h)
    cv2.imshow("Split S {:d}".format(num),s)
    cv2.imshow("Split V {:d}".format(num),v)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray {:d}".format(num),gray)

model = keras.models.load_model('models/model3')
scaler = pickle.load(open('models/scaler3.pkl','rb'))
colorDetector = acd.AutoColorDetector()

parser = argparse.ArgumentParser()
parser.add_argument("building", help="Path to a subfolder in test_imgs")
args = parser.parse_args()

prefix = "test_imgs/{}/".format(args.building)

sample_file_name = prefix + "bld1_0.jpg"
if os.path.isfile(sample_file_name):
    sample = cv2.imread(sample_file_name, cv2.COLOR_BGR2RGB)
    sample = resizeImg(sample, 600, 800)


    labels1, masks1, fts1 = colorDetector.detectBuildingColor(sample)

    cap = cv2.VideoCapture('health-full.mp4')
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
            num_match = 0

            for i in range(len(masks1)):
                for j in range(len(masks2)):

                    if i == j:
                        continue

                    testFts = np.hstack((fts1[i], fts2[j]))
                    pr = predict(testFts)

                    if pr:

                        red = np.zeros((masks2[j].shape[0], masks2[j].shape[1], 3), dtype=np.uint8)
                        red[masks2[j] == 255] = (0,0,255)
                        # red = cv2.resize(red, (2*red.shape[1], 2*red.shape[0]))
                        # copy = cv2.addWeighted(copy, 1, red, 0.4, 0.0)
                        # img1 = cv2.bitwise_and(sample,sample,mask = masks1[i])
                        # img2 = cv2.bitwise_and(current,current,mask = masks2[j])
                        #
                        # cv2.imshow("img1", img1)
                        # cv2.imshow("img2", img2)
                        #
                        # showHSV(img1,1)
                        # showHSV(img2,2)
                        #
                        # cv2.waitKey()
                        num_match += 1



            # if num_match > 3:
            #     copy = cv2.putText(copy, 'Match', (50,50), cv2.FONT_HERSHEY_SIMPLEX, \
            #                        1, (0,255,0), 2, cv2.LINE_AA)
            # else:
            copy = cv2.putText(copy, 'No match', (50,50), cv2.FONT_HERSHEY_SIMPLEX, \
                                   1, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow("reference", sample)
            cv2.imshow("observation", copy)
            cv2.waitKey()

        cv2.imshow("reference", sample)
        cv2.imshow("observation", copy)
        key = cv2.waitKey()
