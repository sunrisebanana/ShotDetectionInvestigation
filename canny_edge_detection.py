#!/usr/bin/env python

'''
Video histogram sample to show live histogram of video
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

# built-in modules
import sys

# local modules
# import video

class App():

    def set_scale(self, val):
        self.hist_scale = val

    def run(self):
        cam = cv.VideoCapture("test.mp4")

        while True:
            _flag, frame = cam.read()
            cv.imshow('camera', frame)

            edges = cv.Canny(frame,100,200)
            cv.imshow("edges", edges)

            ch = cv.waitKey(1)
            if ch == 27:
                break

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()