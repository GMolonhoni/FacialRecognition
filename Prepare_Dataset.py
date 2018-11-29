#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from mtcnn.mtcnn import MTCNN
import sys
import _thread
import time
from threading import Thread
import os
import numpy as np


datasetPath = 'Raw_Dataset'

print("opening mtcnn")
detector = MTCNN()



 
for person in os.listdir(datasetPath):
    print(person)
    for image in os.listdir(datasetPath + '/' + person):
        img = cv2.imread(datasetPath + '/' + person + '/' + image)
        result = detector.detect_faces(img)    
    #Use MTCNN to detect faces
        if result != [] and img is not None:
            for face in result:
                bounding_box = face['box']
                keypoints = face['keypoints']
    
                cv2.rectangle(img,
                   (bounding_box[0], bounding_box[1]),
                   (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                   (0,155,255),
                    2) 
                print(img.shape)
                print(bounding_box[0])                
                print(bounding_box[1])
                print(bounding_box[2])
                print(bounding_box[3])
                print(bounding_box[2]+bounding_box[0])
                print(bounding_box[3]+bounding_box[1])
				
                if 	bounding_box[2] > 160 and bounding_box[3] > 160:
                    frame2 = img[bounding_box[1]+2:(bounding_box[1] + bounding_box[3])-2, bounding_box[0]+2:(bounding_box[0]+bounding_box[2])-2]
                    img2 = cv2.resize(frame2,(160,160))
                    cv2.imwrite('Dataset' + '/' + person + '/' + image ,img2)

print("Finished")