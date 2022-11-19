#!/usr/bin/env python

import numpy as np
import skimage
import cv2 as cv2
from matplotlib import pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
import os
import sys
import json
import pickle
sys.path.append("/home/hojat/Desktop/building_detection")
import autoColorDetection as acd


user_contour = [[]]

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

def mouse_click(event, x, y, flags, param):
    global user_contour

    if event == cv2.EVENT_LBUTTONDOWN:
        user_contour[0].append([x,y])

def defineWalls(img):
    global user_contour

    imgCopy = img.copy()
    imgCopy = resizeImg(imgCopy, 480, 640)

    cv2.namedWindow("define walls")
    cv2.setMouseCallback("define walls", mouse_click)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)*255

    key = ''
    while key != ord('q'):
        imgCopy1 = imgCopy.copy()
        imgCopy1 = resizeImg(imgCopy1, 480, 640)

        if key == ord('d'):
            cv2.drawContours(imgCopy, np.array(user_contour), -1, (0,0,0), thickness=-1)
            user_contour = [ [[int(cnt[0] * img.shape[1] / imgCopy.shape[1]),
                             int(cnt[1] * img.shape[0] / imgCopy.shape[0])] for cnt in user_contour[0]] ]
            cv2.drawContours(mask, np.array(user_contour), -1, 255, thickness=-1)

            user_contour = [[]]

        if len(user_contour[0]) > 0:
            cv2.drawContours(imgCopy1, np.array(user_contour), -1, (0,0,255), thickness=2)

        cv2.imshow("define walls", imgCopy1)

        key = cv2.waitKey(10)

    mask = cv2.bitwise_not(mask)
    mask[0,:] = 0
    mask[-1,:] = 0
    mask[:,0] = 0
    mask[:,-1] = 0
    # cv2.imshow("define walls", mask)
    # key = cv2.waitKey()

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return 255 - mask


manual_data_file = 'processed_manual.json'

if not os.path.isfile(manual_data_file):

    print('##### Annotating Data #####')

    images = {'blds':[]}

    bld_count = 1
    while True:
        sample_file_name = "raw/bld{:d}_0.jpg".format(bld_count)
        if os.path.isfile(sample_file_name):
            sample = cv2.imread(sample_file_name, cv2.COLOR_BGR2RGB)
            sample = resizeImg(sample, 600, 800)

            mask = defineWalls(sample)

            bld = {'ref':sample.tolist(), 'ref_mask':mask.tolist(), 'observations':[]}
            images['blds'].append(bld)

            bld_count1 = 1
            while True:

                current_file_name = "raw/bld{:d}_{:d}.jpg".format(bld_count, bld_count1)
                if os.path.isfile(current_file_name):
                    current = cv2.imread(current_file_name, cv2.COLOR_BGR2RGB)
                    current = resizeImg(current, 600, 800)

                    mask = defineWalls(current)

                    obs = {'obs':current.tolist(), 'obs_mask':mask.tolist()}
                    bld['observations'].append(obs)

                else:
                    break

                bld_count1 = bld_count1 + 1
        else:
            break

        bld_count = bld_count + 1

    with open('processed_manual.json', 'w') as convert_file:
         convert_file.write(json.dumps(images))

else:

    print('##### Loading Annotated Data #####')

    with open('processed_manual.json') as json_file:
        images = json.load(json_file)

    for c_bld in range(len(images['blds'])):

        bld = images['blds'][c_bld]
        bld['ref'] = np.array(bld['ref'], dtype=np.uint8)
        bld['ref_mask'] = np.array(bld['ref_mask'], dtype=np.uint8)

        blue = np.zeros((bld['ref_mask'].shape[0], bld['ref_mask'].shape[1], 3), dtype=np.uint8)
        blue[bld['ref_mask'] == 255] = (255,0,0)
        copy = cv2.addWeighted(bld['ref'], 1, blue, 0.4, 0.0)

        cv2.imshow('review', copy)
        cv2.waitKey(200)


        for c_obs in range(len(bld['observations'])):

            obs = bld['observations'][c_obs]
            obs['obs'] = np.array(obs['obs'], dtype=np.uint8)
            obs['obs_mask'] = np.array(obs['obs_mask'], dtype=np.uint8)

            blue = np.zeros((obs['obs_mask'].shape[0], obs['obs_mask'].shape[1], 3), dtype=np.uint8)
            blue[obs['obs_mask'] == 255] = (255,0,0)
            copy = cv2.addWeighted(obs['obs'], 1, blue, 0.4, 0.0)

            cv2.imshow('review', copy)
            cv2.waitKey(200)

cv2.destroyAllWindows()

features = []

features_file = 'features.pickle'

if not os.path.isfile(features_file):

    print('##### Extracting Features #####')

    colorDetector = acd.AutoColorDetector()
    fig, ax = plt.subplots(2)

    for c_bld in range(len(images['blds'])):

        print('building number {:d}'.format(c_bld))

        bld = images['blds'][c_bld]

        sample = bld['ref'].copy()
        mask = bld['ref_mask'].copy()
        labels_ref, masks_ref, fts_ref = colorDetector.detectBuildingColor(sample, mask=mask)
        labels_ref_neg, masks_ref_neg, fts_ref_neg = colorDetector.detectBuildingColor(sample, mask=255-mask)

        features.append([])
        features[-1].append({'len_positive':len(masks_ref), 'len_negative':len(masks_ref_neg), 'positive': fts_ref, 'negative': fts_ref_neg})

        if labels_ref.any() != None:

            out = color.label2rgb(labels_ref, sample, kind='overlay', bg_label=0)

            ax[0].imshow(out)
            ax[0].set_axis_off()

        for c_obs in range(len(bld['observations'])):

            print('\tobservation {:d}'.format(c_obs))

            obs = bld['observations'][c_obs]

            current = obs['obs'].copy()
            mask = obs['obs_mask'].copy()
            labels_obs, masks_obs, fts_obs = colorDetector.detectBuildingColor(current, mask=mask)
            labels_obs_neg, masks_obs_neg, fts_obs_neg = colorDetector.detectBuildingColor(current, mask=255-mask)

            features[-1].append({'len_positive':len(masks_obs), 'len_negative':len(fts_obs_neg), 'positive': fts_obs, 'negative': fts_obs_neg})

            if labels_obs.any() != None:

                out = color.label2rgb(labels_obs, current, kind='overlay', bg_label=0)

                ax[1].imshow(out)
                ax[1].set_axis_off()

                # plt.show()

    with open('features.pickle', 'wb') as output_file:
        pickle.dump(features, output_file)

else:

    print('##### Loading Features #####')

    with open('features.pickle', 'rb') as input_file:
        features = pickle.load(input_file)


print('##### Pairing Features #####')

data = []

for i in range(len(features)):

    for j in range(len(features[i])):

        for k in range(j,len(features[i])):

            if j == k: continue

            fts_1_pos = features[i][j]['positive']
            fts_1_neg = features[i][j]['negative']
            fts_2_pos = features[i][k]['positive']
            fts_2_neg = features[i][k]['negative']

            len_1_pos = features[i][j]['len_positive']
            len_1_neg = features[i][j]['len_negative']
            len_2_pos = features[i][k]['len_positive']
            len_2_neg = features[i][k]['len_negative']

            print('Adding {:d} positive samples.'.format(len_1_pos*len_2_pos))

            for i1 in range(len_1_pos):
                for i2 in range(len_2_pos):
                    d1 = fts_1_pos[i1]
                    d2 = fts_2_pos[i2]

                    if d1 is not None and d2 is not None:
                        trueFts = np.hstack((d1, d2))
                        data.append( np.hstack( (trueFts,np.array([1])) ))

            print('Adding {:d} negative samples.'.format(len_1_neg*len_2_neg + \
                                                              len_1_pos*len_2_neg +
                                                              len_1_neg*len_2_pos))

            for i1 in range(len_1_neg):
                for i2 in range(len_2_neg):
                    d1 = fts_1_neg[i1]
                    d2 = fts_2_neg[i2]

                    if d1 is not None and d2 is not None:
                        falseFts = np.hstack((d1, d2))
                        data.append( np.hstack( (falseFts,np.array([0])) ))

            for i1 in range(len_1_pos):
                for i2 in range(len_2_neg):
                    d1 = fts_1_pos[i1]
                    d2 = fts_2_neg[i2]

                    if d1 is not None and d2 is not None:
                        falseFts = np.hstack((d1, d2))
                        data.append( np.hstack( (falseFts,np.array([0])) ))

            for i1 in range(len_1_neg):
                for i2 in range(len_2_pos):
                    d1 = fts_1_neg[i1]
                    d2 = fts_2_pos[i2]

                    if d1 is not None and d2 is not None:
                        falseFts = np.hstack((d1, d2))
                        data.append( np.hstack( (falseFts,np.array([0])) ))

np.savetxt("features/data.csv", data, delimiter=",")
