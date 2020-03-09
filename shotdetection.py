#!/usr/bin/env python

'''
Video histogram sample to show live histogram of video
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import copy
import numpy as np
import cv2 as cv
import time

# built-in modules
import sys

def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

class App():

    
    def run(self):
        hsv_map = np.zeros((180, 256, 3), np.uint8)
        h, s = np.indices(hsv_map.shape[:2])
        hsv_map[:,:,0] = h
        hsv_map[:,:,1] = s
        hsv_map[:,:,2] = 255
        hsv_map = cv.cvtColor(hsv_map, cv.COLOR_HSV2BGR)
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,20)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        cv.imshow('hsv_map', hsv_map)

        cv.namedWindow('hist', 0)
        self.hist_scale = 10

        histChange, edgeChange = 0, 1
        first = True
        try:
            fn = sys.argv[1]
        except:
            fn = 0
        cam = cv.VideoCapture("test.mp4")
        f = 0

        successesHist, falsePositivesHist, successesEdge, falsePositivesEdge = 0,0,0,0
        with open('cuts.txt') as file:
            cuts = file.read().split(',')
            cuts = list(map(int, cuts))
        while True:
            try:
                _flag, frame = cam.read()
                cv.imshow('camera', frame)
            except:
                break

            small = cv.pyrDown(frame)

            

            hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
            
            dark = hsv[...,2] < 32
            hsv[dark] = 0
            if not first:
                prevFrameH = copy.deepcopy(h)
                h = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                histChange = cv.compareHist(prevFrameH, h, method=3)
                # histChange = ssim(prevFrameH, h)
                # histChange = cv.EMD(h, prevFrameH, distType=0)
                
                prevEdges = copy.deepcopy(edges)
                edges = cv.Canny(frame,150,200)
                #edgeChange = cv.compareHist(prevEdges, edges, method=0)
                edgeChange = ssim(prevEdges, edges)
                cv.putText(edges, str(f) + " " + str(edgeChange), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
                cv.imshow("edges", edges)
            else:
                h = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                edges = cv.Canny(frame,100,200)
                cv.imshow("edges", edges)

            h = np.clip(h*0.005*self.hist_scale, 0, 1)
            vis = hsv_map*h[:,:,np.newaxis] / 255.0
            img = cv.putText(vis, str(histChange), (10,500), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255), 2, 0)
            
            cv.putText(img, str(f) + " " + str(histChange), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv.imshow('hist', img)
           
            first = False
            # if (histChange > biggestHistChange):
            #     biggestHistChange = histChange
            #     print(str(f) + " hist " + str(biggestHistChange))
            # if (edgeChange > biggestEdgeChange):
            #     biggestEdgeChange = edgeChange
            #     print(str(f) + " edge " + str(biggestEdgeChange))
            if (histChange > 0.5):
                print(str(f) + " hist " + str(histChange))
                if cuts.__contains__(f):
                    successesHist += 1
                else:
                    falsePositivesHist += 1
            if (edgeChange < 0.7):
                print(str(f) + " edge " + str(edgeChange))
                if cuts.__contains__(f):
                    successesEdge += 1
                else:
                    falsePositivesEdge += 1
            ch = cv.waitKey(1)
            if ch == 27:
                break
            if ch == ord(' '):
                time.sleep(1)
            f += 1
        print('Histogram successes: ' + str(successesHist) + ' out of ' + str(len(cuts)) + ', percentage: ' + str(successesHist / len(cuts)))
        print('Histogram false positives: ' + str(falsePositivesHist))
        print('Canny Edge successes: ' + str(successesEdge) + ' out of ' + str(len(cuts)) + ', percentage: ' + str(successesEdge / len(cuts)))
        print('Canny Edge false positives: ' + str(falsePositivesEdge))


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()