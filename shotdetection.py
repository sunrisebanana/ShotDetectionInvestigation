#!/usr/bin/env python

'''
Video histogram sample to show live histogram of video
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import copy
import numpy as np
import cv2 as cv

# built-in modules
import sys


class App():

    def set_scale(self, val):
        self.hist_scale = val

    def run(self):
        hsv_map = np.zeros((180, 256, 3), np.uint8)
        h, s = np.indices(hsv_map.shape[:2])
        hsv_map[:,:,0] = h
        hsv_map[:,:,1] = s
        hsv_map[:,:,2] = 255
        hsv_map = cv.cvtColor(hsv_map, cv.COLOR_HSV2BGR)
        cv.imshow('hsv_map', hsv_map)

        cv.namedWindow('hist', 0)
        self.hist_scale = 10

        histChange = 0
        first = True
        try:
            fn = sys.argv[1]
        except:
            fn = 0
        cam = cv.VideoCapture("test.mp4")

        while True:
            _flag, frame = cam.read()
            cv.imshow('camera', frame)

            small = cv.pyrDown(frame)

            edges = cv.Canny(frame,100,200)
            cv.imshow("edges", edges)

            hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
            
            dark = hsv[...,2] < 32
            hsv[dark] = 0
            if not first:
                prevFrameH = copy.deepcopy(h)
                h = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                histChange = cv.compareHist(prevFrameH, h, method=0)*100
            else:
                h = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            h = np.clip(h*0.005*self.hist_scale, 0, 1)
            vis = hsv_map*h[:,:,np.newaxis] / 255.0
            img = cv.putText(vis, str(histChange), (10,500), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255), 2, 0)
            cv.imshow('hist', img)
           
            first = False
            ch = cv.waitKey(1)
            if ch == 27:
                break

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()