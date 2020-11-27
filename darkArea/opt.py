#!/usr/bin/env python

import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from scipy.signal import find_peaks
import utility as utl

def preProccessSegment(segment, darkSegmentThresh):

    img = cv.cvtColor(segment, cv.COLOR_BGR2HSV)[:,:,2]
    minVal = np.min(img)
    maxVal = np.max(img)
    diff = maxVal - minVal
    # print 'mean', np.mean(img)
    offset = darkSegmentThresh * diff
    ret,imgThresholded = cv.threshold(img, minVal+offset, 255, cv.THRESH_BINARY_INV)
    imageThresholded = cv.erode(imgThresholded, np.ones((2,2),np.uint8))

    return imageThresholded

def process(num):
    img = cv.imread("dataset/w{:d}.jpg".format(num))

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lap = cv.Laplacian(imgGray, cv.CV_16S, ksize=3)
    abs = cv.convertScaleAbs(lap)

    vertical = cv.reduce(abs, 1, cv.REDUCE_AVG)
    horizontal = cv.reduce(abs, 0, cv.REDUCE_AVG)

    vertical = vertical * ( float(lap.shape[1]) / max(1, vertical.max()) )
    horizontal = horizontal * ( float(lap.shape[0]) / max(1, horizontal.max()) )

    vpeaks = find_peaks(vertical.flatten(), prominence=15, distance=5)
    hpeaks = find_peaks(horizontal.flatten(), prominence=15, distance=5)

    vpeaks = np.insert(vpeaks[0], 0, 0)
    vpeaks = np.append(vpeaks, img.shape[0])

    hpeaks = np.insert(hpeaks[0], 0, 0)
    hpeaks = np.append(hpeaks, img.shape[1])

    costs = np.zeros(100, dtype=np.float32)
    test = np.zeros(100, dtype=np.float32)
    imgs = []

    segs = np.zeros(imgGray.shape, dtype=np.uint8)
    segs1 = np.zeros(imgGray.shape, dtype=np.uint8)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    imgArea = imgGray.shape[0] * imgGray.shape[1]
    minmean = 255
    minrect = ()

    for h, hpeak in enumerate(hpeaks):
        for v, vpeak in enumerate(vpeaks):

            if h == len(hpeaks)-1 or v == len(vpeaks)-1: continue
            rect = (hpeaks[h], vpeaks[v], hpeaks[h+1] - hpeaks[h], vpeaks[v+1] - vpeaks[v])

            mean = np.mean(imgHSV[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], 2])

            if mean < minmean:
                if rect[2]*rect[3] > 0.01 * imgArea:
                    minmean = mean
                    minrect = rect

            cv.rectangle(segs, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), mean, -1)


    cv.rectangle(segs1, (minrect[0], minrect[1]), (minrect[0]+minrect[2], minrect[1]+minrect[3]), 255, -1)

    fig, ax = plt.subplots()
    ax.imshow(segs, cmap='gray')
    ax.set_axis_off()
    plt.savefig('result/d{:d}.png'.format(num))


    for i in range(100):

        imgThresh = preProccessSegment(img, i/100.0)

        if i in [10,30,50,70,90]:
            imgs.append(imgThresh)

        cost = 0.0
        cs = []

        for h, hpeak in enumerate(hpeaks):
            for v, vpeak in enumerate(vpeaks):

                if h == len(hpeaks)-1 or v == len(vpeaks)-1: continue
                rect = (hpeaks[h], vpeaks[v], hpeaks[h+1] - hpeaks[h], vpeaks[v+1] - vpeaks[v])

                mask = np.zeros(imgThresh.shape, dtype=np.uint8)
                cv.rectangle(mask, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, -1)

                res = cv.bitwise_and(imgThresh, imgThresh, mask = mask)
                # c = np.float32(np.count_nonzero(res)) / np.float32(np.count_nonzero(mask))
                c = np.abs( ( np.float32(np.count_nonzero(res)) / np.float32(np.count_nonzero(mask)) )- 0.5)

                cs.append( c )

                # imgcopy = img.copy()
                # cv.rectangle(imgcopy, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 1)
                # cv.imshow("rects", res)
                # cv.waitKey(3000)

        # cost = np.sum(cs) / len(cs) - 0.*np.abs( ( np.float32(np.count_nonzero(imgThresh)) / np.float32(imgThresh.shape[0]*imgThresh.shape[1]) )- 0.5)

        diff = imgThresh.astype(np.int16) - segs1.astype(np.int16)
        idxs = np.where(diff > 0)
        diff[idxs] = 0.3*diff[idxs]
        cost = np.sum(np.abs(diff)).astype(np.float32) / imgArea
        # test[i] = - np.abs( ( np.float32(np.count_nonzero(imgThresh)) / np.float32(imgThresh.shape[0]*imgThresh.shape[1]) )- 0.5)
        # if np.std(cs) < 0.03:
        #     cost = 0

        # print i, cost, 5*np.std(cs)
        # print cost

        # for peak in hpeaks:
        #     cv.line(imgThresh, (peak,0), (peak, imgThresh.shape[0]), 255, 1)
        # for peak in vpeaks:
        #     cv.line(imgThresh, (0,peak), (imgThresh.shape[1], peak), 255, 1)

        # cv.imshow("abs", np.abs(imgThresh.astype(np.int16) - segs1.astype(np.int16)).astype(np.uint8))
        # cv.imshow("thresh", imgThresh)
        # cv.waitKey()

        costs[i] = cost

    min = np.argmin(costs)
    imgThresh = preProccessSegment(img, min/100.0)
    # print min

    imgCopy = img.copy()
    imgCopy = cv.resize(imgCopy, (5*imgCopy.shape[1], 5*imgCopy.shape[0]) )
    imageThresholded = preProccessSegment(imgCopy, min/100.0)

    size1 = 6
    size2 = 10
    element1 = cv.getStructuringElement(cv.MORPH_RECT, (2*size1+1, 2*size1+1), (size1, size1))
    element2 = cv.getStructuringElement(cv.MORPH_RECT, (2*size2+1, 2*size2+1), (size2, size2))
    imageThresholded = cv.dilate(imageThresholded, element1)
    imageThresholded = cv.erode(imageThresholded, element2)

    _, contours, hierarchy = cv.findContours(imageThresholded, cv.RETR_CCOMP , cv.CHAIN_APPROX_TC89_KCOS)
    if len(contours)!=0:

        maxArea = -1
        i = None
        for idx, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            if area<30 or len(cnt)<=3:
                continue
            if area>maxArea:
                maxArea = area
                i = idx
        if i!=None:
            poly = cv.approxPolyDP(contours[i],25, True)
            poly = utl.reconstructRect(poly, True)
            cv.drawContours(imgCopy, [poly], 0, (0,255,0), 2)

    fig, ax = plt.subplots()
    # cv.rectangle(imgCopy, (minrect[0], minrect[1]), (minrect[0]+minrect[2], minrect[1]+minrect[3]), (0,0,255))
    ax.imshow(imgCopy)
    ax.set_axis_off()
    plt.savefig('result/o{:d}.png'.format(num))
    # ax[1].imshow(imageThresholded, cmap='gray')

    # fig, ax = plt.subplots()
    # ax.plot(test)
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.imshow(img, cmap='gray')
    # plt.show()

    fig, ax = plt.subplots()
    ax.imshow(lap, cmap='gray')
    ax.set_axis_off()
    # plt.show()
    plt.savefig('result/lap{:d}.png'.format(num))

    fig, ax = plt.subplots()
    ax.imshow(lap, cmap='gray')
    ax.plot(range(lap.shape[1]), horizontal.ravel()/3, '-', linewidth=2, color='r')
    ax.plot(vertical/3, range(lap.shape[0]), '-', linewidth=2, color='g')
    # plt.show()
    plt.savefig('result/dist{:d}.png'.format(num))

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_axis_off()
    for peak in hpeaks:
        ax.axvline(x=peak, ymin=0.0, ymax=1.0, color='r', linewidth=2)

    for peak in vpeaks:
        ax.axhline(y=peak, xmin=0.0, xmax=1.0, color='g', linewidth=2)

    plt.savefig('result/segs{:d}.png'.format(num))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    # ax.set_ylim(-0.2, 1.0)

    ax.plot(costs, linewidth=2, color='blue')
    maxcost = np.max(costs)

    cnt = 0

    for i in [10,30,50,70,90]:
        if i == 10:
            cnt = cnt + 1
            continue

        ax.axvline(x=i, ymin=0.0, ymax=1.0, color='black', linewidth=1, linestyle='--')
        binImg = imgs[cnt]
        binImg = cv.cvtColor(binImg, cv.COLOR_GRAY2BGR)
        imagebox = OffsetImage(binImg, zoom=0.3)
        ab = AnnotationBbox(imagebox, (i, 0.8*maxcost))
        ax.add_artist(ab)

        cnt = cnt + 1

    ax.axvline(x=min, ymin=0.0, ymax=1.0, color='red', linewidth=2, linestyle='--')
    binImg = imageThresholded
    binImg = cv.resize(imageThresholded, (imgCopy.shape[1]/5, imgCopy.shape[0]/5) )
    binImg = cv.cvtColor(binImg, cv.COLOR_GRAY2BGR)
    imagebox = OffsetImage(binImg, zoom=0.45)
    ab = AnnotationBbox(imagebox, (min, 0.8*maxcost))
    ax.add_artist(ab)

    plt.xlabel("Threshold Value")
    plt.ylabel("Cost Function")
    plt.draw()
    # plt.show()
    plt.savefig('result/r{:d}.png'.format(num))

    # cv.imshow("lap", abs)
    # cv.imshow("thresh", imageThresholded)
    # cv.imshow("poly", imgCopy)
    # cv.waitKey()


for i in range(1,84):
    print i, "/", 84
    process(i)
