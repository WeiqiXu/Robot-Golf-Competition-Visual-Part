# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:05:06 2018

@author: meringue
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.png',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,16,
                            param1=200,param2=16,minRadius=1,maxRadius=48)
                            
        
if circles is not None:
    circles = np.uint16(np.around(circles[0, ]))
    for i in circles[:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(100,255,100),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,100,255),3)
        
        print i
else:
    print "no circle."


#plt.imshow(cimg, cmap="gray")
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
#cv2.destroyAllWindows()


# print type(circles)
# print circles.shape
# print circles[0][1]
