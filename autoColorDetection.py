#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import time
from matplotlib import pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
from skimage.feature import greycomatrix, greycoprops
from skimage.exposure import rescale_intensity

class AutoColorDetector:

    def __init__(self):
        # assert method == 0 or method == 1

        # self._method = method
        # self._deviationFactor = deviationFactor
        # self._segmentThreshold = segmentThreshold
        # self._peakNeighbourhood = peakNeighbourhood
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255


    def cut(self, img, labels, thresh):
        g = graph.rag_mean_color(img, labels)
        labels1 = graph.cut_threshold(labels, g, thresh)

        return labels1

    def segmentRAG(self,img, thresh, mask=None):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels1 = segmentation.slic(img, compactness=30, n_segments=600, mask=mask)

        labels2 = self.cut(img, labels1, thresh)

        return labels2

    def filterSegments(self, image, labels, segThresh):

        maxArea = 0
        maxIdx = None

        new_labels = np.zeros(labels.shape, np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        segs = []
        masks = []

        unique, counts = np.unique(labels, return_counts=True)
        for i, unq in enumerate(unique):

            if unq == 0:
                continue

            if counts[i] > 3*segThresh:
                new_labels[labels==unq] = unq
                mask = np.zeros(labels.shape, np.uint8)
                mask[labels==unq] = 255

                cut = cv2.bitwise_and(gray,gray,mask = mask)

                masks.append(mask)
                segs.append(cut)


        return new_labels, segs, masks

    def textureFeatures(self, segs):

        fts = np.zeros((len(segs), 6))

        for i, seg in enumerate(segs):
            rescaled = rescale_intensity(seg, out_range=(0, 20))
            glcm = greycomatrix(rescaled.astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=21)

            fts[i,0] = greycoprops(glcm, prop='contrast').mean()
            fts[i,1] = greycoprops(glcm, prop='dissimilarity').mean()
            fts[i,2] = greycoprops(glcm, prop='homogeneity').mean()
            fts[i,3] = greycoprops(glcm, prop='ASM').mean()
            fts[i,4] = greycoprops(glcm, prop='energy').mean()
            fts[i,5] = greycoprops(glcm, prop='correlation').mean()

            # cv2.imshow("seg",seg)
            # cv2.waitKey()

            # print fts[i]

        return fts

    def shapeFeatures(self, masks):

        fts = np.zeros((len(masks), 7))

        for i, mask in enumerate(masks):
            fts[i] = cv2.HuMoments(cv2.moments(mask)).flatten()

        return fts

    def colorFeatures(self, segs, masks):

        fts = np.zeros((len(segs), 2))

        for i, seg in enumerate(segs):
            fts[i,0], fts[i,1] = cv2.meanStdDev(seg, mask=masks[i])

        return fts

    def detectBuildingColor(self, image, mask=None):

        assert image.dtype == np.uint8 and len(image.shape) == 3

        blur = cv2.bilateralFilter(image,21,75,75)

        area = 0
        thresh = 5
        labels = None

        labels = self.segmentRAG(blur, thresh, mask=mask)
        labels, segs, masks = self.filterSegments(image, labels, image.shape[0]*image.shape[1] / 600)

        tx_fts = self.textureFeatures(segs)
        # shp_fts = self.shapeFeatures(masks)
        clr_fts = self.colorFeatures(segs, masks)

        fts = np.hstack((tx_fts, clr_fts))


        labels = self.cut(image, labels, 8)
        labels = self.cut(image, labels, 12)

        return labels, masks, fts

    def segmentIdx(self, masks, p):

        idx = None
        for i, mask in enumerate(masks):
            if mask[p[1], p[0]] != 0:
                idx = i
        return idx

    def getFts(self, masks, fts, p):

        idx = self.segmentIdx(masks, p)

        if idx:
            return fts[idx], idx
        else:
            return None, None
