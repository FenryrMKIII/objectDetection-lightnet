#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# inspired from https://github.com/decanbay/YOLOv3-Calculate-Anchor-Boxes/blob/master/YOLOv3_get_anchors.py

import argparse
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set() 
from pathlib import Path
from sklearn.cluster import KMeans

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute YoloV3 anchor boxes based on dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )



    #parser.add_argument('-i', ,'--img', help='Path to image files', default=None, required=True)
    parser.add_argument('-a', '--anno', help='Path to annotations file', default=None, type=str, required=True)
    parser.add_argument('-r', '--resolution', help='Input resolution that will be used in YoloV3', type=int, default=None, required=True)
    parser.add_argument('-k', '--klusters', help='Number of k-means (k clusters) to be computed', type=int, default=None, required=True)
    args = parser.parse_args()
# extract relative width & height data of training set
annoPath = [anno for anno in Path(args.anno).iterdir() if anno.suffix == '.txt']
# do not know array size beforehand
# so create list then trasform to array
# this is faster than extending an array
# faster way could be pre-allocating a huge array and resizing at the end
bboxData = []
for i, anno in enumerate(annoPath):
    for line in anno.open().readlines() :
        bboxData.append((float(line.split()[3]),  # store relative width
                        float(line.split()[4]))) # and height
bboxData = np.array(bboxData)

# compute K-means (yolo anchors) based on training set data
kmeans3 = KMeans(n_clusters=args.klusters)
kmeans3.fit(bboxData)
y_kmeans3 = kmeans3.predict(bboxData)

yolo_anchor = kmeans3.cluster_centers_

# vizualize
plt.scatter(bboxData[:, 0], bboxData[:, 1], c=y_kmeans3, s=2, cmap='viridis')
plt.scatter(yolo_anchor[:, 0], yolo_anchor[:, 1], c='red', s=50);
yoloV3anchorsCustom = yolo_anchor
yoloV3anchorsCustom[:, 0] =yolo_anchor[:, 0] * args.resolution
yoloV3anchorsCustom[:, 1] =yolo_anchor[:, 1] * args.resolution
yoloV3anchorsCustom = np.rint(yoloV3anchorsCustom)
yoloV3anchors = np.array([(116, 90), (156, 198), (373, 326), (30, 61), (62, 45), (59, 119), (10, 13), (16, 30), (33, 23)])
fig, ax = plt.subplots()
for ind in range(args.klusters):
    rectangle= plt.Rectangle((args.resolution/2-yoloV3anchorsCustom[ind,0]/2,
                              args.resolution/2-yoloV3anchorsCustom[ind,1]/2), 
                              yoloV3anchorsCustom[ind,0],
                              yoloV3anchorsCustom[ind,1] , 
                              fc='b',edgecolor='b',fill = None)
    ax.add_patch(rectangle)
    rectangle2= plt.Rectangle((args.resolution/2-yoloV3anchors[ind,0]/2,
                              args.resolution/2-yoloV3anchors[ind,1]/2), 
                              yoloV3anchors[ind,0],
                              yoloV3anchors[ind,1] , 
                              fc='r',edgecolor='r',fill = None)
    ax.add_patch(rectangle2)

# from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor='orange', edgecolor='r',
#                          label='Color Patch')]
# ax.legend(handles=legend_elements)
ax.set_aspect(1.0)
plt.axis([0,args.resolution,0,args.resolution])
plt.show()
yoloV3anchors.sort(axis=0)
print("Your custom anchor boxes are {}".format(yoloV3anchors))

F = open("YOLOV3_Anchors.txt", "w")
F.write("{}".format(yoloV3anchors))
F.close() 