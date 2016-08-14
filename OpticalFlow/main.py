#!/usr/bin/python

import cv2
import numpy as np

def track(arg):
    frames = []

    if isinstance(arg,list):
        # sequence of imgs
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
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            frame = cv2.fastNlMeansDenoisingColored(frame)
            frame = cv2.pyrMeanShiftFiltering(frame,5,5)
            cv2.imwrite(str(i) + '.png',frame)
            frames.append(frame)
            print(i)
            i = i+1

        cap.release()

    prv = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frames[0])
    hsv[...,1] = 255

    for frame in frames:
        nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prv = nxt

    cv2.destroyAllWindows()

if __name__ == "__main__":
    track([str(i)+'.png' for i in range(176)])

#     if len(sys.argv) < 2:
#         print "USAGE : {} <filename>".format(sys.argv[0])
#     else:
#         track(sys.argv[1])
