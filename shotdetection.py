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
from matplotlib import pyplot as plt
import warnings

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

## source: https://github.com/yonatankatz/edge-change-ratio-example/blob/master/ecr.py
def ECR(edge, edge2, width, height, dilate_rate = 5):
    safe_div = lambda x,y: 0 if y == 0 else x / y
    
    dilated = cv.dilate(edge, np.ones((dilate_rate, dilate_rate)))
    dilated2 = cv.dilate(edge2, np.ones((dilate_rate, dilate_rate)))
    inverted = (255 - dilated)
    inverted2 = (255 - dilated2)
    log_and1 = (edge2 & inverted)
    log_and2 = (edge & inverted2)
    pixels_sum_new = np.sum(edge)
    pixels_sum_old = np.sum(edge2)
    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)
    return max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))
    
class App():

    
    def run(self):
        warnings.filterwarnings("ignore")
        plt.show()
        fig = plt.figure()
        tests = [
            "regtest",
            "gametest",
            "gametest2",
            "gametest3",
            "gametest4",
        ]
        testtype1 = "Bhattarcharyya"
        testtype2 = "ECR"
        for test in tests:
            file = open(test + "results" + testtype1 + testtype2 + ".txt", "w")
            file.close()
            for threshold in [x/20 for x in range(0, 20)]:
            # for threshold in [0.4]:
                font                   = cv.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,20)
                fontScale              = 1
                fontColor              = (255,255,255)
                lineType               = 2

                histChange, edgeChange = 0, 0
                first = True
                cam = cv.VideoCapture(test + ".mp4")
                f = 0

                successesHist, falsePositivesHist, successesEdge, falsePositivesEdge = 0,0,0,0
                with open(test + 'cuts.txt') as file:
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
                        h = cv.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        histChange = cv.compareHist(prevFrameH, h, method=3)
                        # histChange = ssim(prevFrameH, h)
                        # histChange = cv.EMD(h, prevFrameH, distType=0)
                        
                        prevFrame = copy.deepcopy(currFrame)
                        currFrame = copy.deepcopy(small)
                        gray_image = cv.cvtColor(currFrame, cv.COLOR_BGR2GRAY)
                        edges = cv.Canny(gray_image, 100, 200)
                        gray_image2 = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
                        edge2 = cv.Canny(gray_image2, 100, 200)
                        edgeChange = ECR(edges, edge2, prevFrame.shape[1], prevFrame.shape[0], )

                        # prevEdges = copy.deepcopy(edges)
                        # edges = cv.Canny(frame,150,200)
                        # edgeChange = ssim(prevEdges, edges)

                        # edgeChange = cv.compareHist(prevEdges, edges, method=0)
                        
                        cv.putText(edges, str(f) + " " + str(edgeChange), 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
                        cv.imshow("edges", edges)
                    else:
                        h = cv.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        edges = cv.Canny(small,0,200)
                        currFrame = copy.deepcopy(small)
                        cv.imshow("edges", edges)

                    # h = np.clip(h*0.005*self.hist_scale, 0, 1)
                    # vis = hsv_map*h[:,:,np.newaxis] / 255.0
                    # img = cv.putText(h, str(histChange), (10,500), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255), 2, 0)
                    
                    # cv.putText(img, str(f) + " " + str(histChange), 
                    #     bottomLeftCornerOfText, 
                    #     font, 
                    #     fontScale,
                    #     fontColor,
                    #     lineType)
                    # cv.imshow('hist', img)
                    
                    
                    ax = fig.add_subplot(1,1,1)

                    features = []
                    chans = cv.split(small)
                    colors = ("b", "g", "r")
                    # loop over the image channels
                    for line in ax.lines:
                        ax.lines.pop(0)
                    for (chan, color) in zip(chans, colors):
                        # create a histogram for the current channel and
                        # concatenate the resulting histograms for each
                        # channel
                        hist = cv.calcHist([chan], [0], None, [8], [0, 256])
                        features.extend(hist)
                        # plot the histogram
                        
                        ax.plot(hist, color = color)
                        plt.xlim([0, 7])
                    # plt.pause(0.000000000001)
                    first = False
                    # if (histChange > biggestHistChange):
                    #     biggestHistChange = histChange
                    #     print(str(f) + " hist " + str(biggestHistChange))
                    # if (edgeChange > biggestEdgeChange):
                    #     biggestEdgeChange = edgeChange
                    #     print(str(f) + " edge " + str(biggestEdgeChange))
                    if (histChange > threshold):
                        print(str(f) + " hist " + str(histChange))
                        if cuts.__contains__(f):
                            successesHist += 1
                        else:
                            falsePositivesHist += 1
                    if (edgeChange > threshold):
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
                file = open(test + "results" + testtype1 + testtype2 + ".txt","a+")
                file.write('Threshold: ' + str(threshold) + ', Histogram comparison method:' + testtype1 + ', Canny edge comparison method:' + testtype2 + '\n')
                file.write('Histogram successes: ' + str(successesHist) + ' out of ' + str(len(cuts)) + ', percentage: ' + str(successesHist / len(cuts)) + '\n')
                file.write('Histogram false positives: ' + str(falsePositivesHist) + '\n')
                file.write('Canny Edge successes: ' + str(successesEdge) + ' out of ' + str(len(cuts)) + ', percentage: ' + str(successesEdge / len(cuts)) + '\n')
                file.write('Canny Edge false positives: ' + str(falsePositivesEdge) + '\n\n\n')
        


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()