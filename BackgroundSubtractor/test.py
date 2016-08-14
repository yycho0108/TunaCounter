#!/usr/bin/python

import cv2
import numpy as np
import sys
import time

cv2.ocl.setUseOpenCL(False) #prevent weird bugs

#fgbg = cv2.createBackgroundSubtractorKNN(15)
fgbg = cv2.createBackgroundSubtractorMOG2()

def track(arg):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    frames = []

    cv2.namedWindow('frame')
    cv2.namedWindow('fgmask')

    if isinstance(arg,list):
        # series of images
        for frame in arg:
            frames.append(cv2.imread(frame))
    elif isinstance(arg,str):
        # video file
        cap = cv2.VideoCapture(arg)
        i = 0
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            frame = cv2.fastNlMeansDenoisingColored(frame)
            frame = cv2.pyrMeanShiftFiltering(frame,5,5)
            cv2.imwrite(str(i) + '.png',frame)
            frames.append(frame)
            print(i)
            i = i+1

    print "READ COMPLETE : {} frames".format(len(frames))

    while cv2.waitKey(0) != 27:
        pass

    for frame in frames:
        fgmask = fgbg.apply(frame)
        cv2.imshow('frame', cv2.resize(frame,(0,0),fx=2,fy=2))
        cv2.imshow('fgmask', cv2.resize(fgmask,(0,0),fx=2,fy=2))
        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    track([str(i)+'.png' for i in range(176)])
    #if len(sys.argv) < 2:
    #    print "USAGE : {} <filename>".format(sys.argv[0])
    #else:
    #    track(sys.argv[1])

# IMG = sys.argv[1]
# while cv2.waitKey() != 27:
#     orig = cv2.imread(IMG_FILE)
#     frame = cv2.resize(orig,(0,0),fx=0.125,fy=0.125)
# 
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# 
#     fgmask = fgbg.apply(frame)
# 
#     cv2.imshow('frame',frame)
#     cv2.imshow('fgmask',fgmask)
# 
# cv2.destroyAllWindows()