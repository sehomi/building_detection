#!/usr/bin/env python

import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

M = np.array( [ [904.041947 , 0.        , 480.879667, 0],
                [0.         , 906.260575, 343.293266, 0],
                [0.         , 0.        , 1.        , 0]] )
z0 = 2000
c1 = 0
c2 = 0
update = True

def on_z0(val):
    global z0, update
    z0 = val
    update = True

def on_c1(val):
    global c1, update
    c1 = (val-200)/100.0
    update = True

def on_c2(val):
    global c2, update
    c2 = (val-200)/100.0
    update = True

def process(imgDir):
    img = cv.imread(imgDir)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    canny = cv.Canny(img, 50, 200, None, 3)

    linesP = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, 20, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)

    cv.imshow("test1", cv.resize(img, (img.shape[1]/2, img.shape[0]/2)))
    # cv.imshow("canny", canny)
    cv.waitKey(0)


def project(points, p0):

    ps = []
    f = 905.

    for point in points:

        u = f * point[0] / point[2]
        v = f * point[1] / point[2]


        ps.append([u+p0[0], v+p0[1]])

    return ps

def inverse_project(points, plane, p0):

    Ps = []
    f = 905.

    for point in points:

        u = point[0] - p0[0]
        v = point[1] - p0[1]

        x = plane[0]*u / (f + plane[1]*u + plane[2]*v)
        y = plane[0]*v / (f + plane[1]*u + plane[2]*v)
        z = plane[0]*f / (f + plane[1]*u + plane[2]*v)

        Ps.append([x, y, z])

    return np.array(Ps)

def draw(img):
    global z0, c1, c2, update

    test = [[50, 50], [ img.shape[1] - 50, img.shape[0] - 50],
            [ img.shape[1] - 50, 50], [50, img.shape[0]- 50]]

    while True:

        if update:

            p3D = inverse_project(test, [z0, c1, c2], (img.shape[1]/2, img.shape[0]/2))

            # center1 = (p3D[0] + p3D[1])/2
            # p3D[0] = 1.1*(p3D[0] - center1) + center1
            # p3D[1] = 1.1*(p3D[1] - center1) + center1
            # center2 = (p3D[2] + p3D[3])/2
            # p3D[2] = 1.1*(p3D[2] - center2) + center2
            # p3D[3] = 1.1*(p3D[3] - center2) + center2

            vers1 = np.linspace(p3D[0], p3D[2], num=10)
            hors1 = np.linspace(p3D[0], p3D[3], num=10)
            vers2 = np.linspace(p3D[3], p3D[1], num=10)
            hors2 = np.linspace(p3D[2], p3D[1], num=10)

            # vers1[:,2] = vers1[:,2] + z0 - 2000
            # hors1[:,2] = hors1[:,2] + z0 - 2000
            # vers2[:,2] = vers2[:,2] + z0 - 2000
            # hors2[:,2] = hors2[:,2] + z0 - 2000

            p2D = project(p3D, (img.shape[1]/2, img.shape[0]/2))
            vers12D = project(vers1, (img.shape[1]/2, img.shape[0]/2))
            hors12D = project(hors1, (img.shape[1]/2, img.shape[0]/2))
            vers22D = project(vers2, (img.shape[1]/2, img.shape[0]/2))
            hors22D = project(hors2, (img.shape[1]/2, img.shape[0]/2))


            imgCopy = img.copy()

            for i in range(len(vers12D)):
                p1 = vers12D[i]
                p2 = vers22D[i]

                cv.line(imgCopy, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255),2)

            for i in range(len(hors12D)):
                p1 = hors12D[i]
                p2 = hors22D[i]
                cv.line(imgCopy, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255),2)

            for p in test:
                cv.circle(imgCopy, (int(p[0]), int(p[1])), 2, (255,0,0), 2)

            for p in p2D:
                cv.circle(imgCopy, (int(p[0]), int(p[1])), 2, (0,0,255), 2)

            update = False

        cv.imshow("test", cv.resize(imgCopy, (img.shape[1]/2, img.shape[0]/2)))
        cv.waitKey(30)


img = cv.imread("etrims/33.jpg")
cv.namedWindow("test")
cv.createTrackbar("z0", "test" , np.int(z0), 4000, on_z0)
cv.createTrackbar("c1", "test" , np.int((c1+2)*100), 400, on_c1)
cv.createTrackbar("c2", "test" , np.int((c2+2)*100), 400, on_c2)

process("etrims/33.jpg")
draw(img)
