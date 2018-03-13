import numpy as np
import cv2

img = cv2.imread("stick.png",1)
"""
cv2.imshow("stick", img)
cv2.waitKey(0)
"""

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgBin = cv2.inRange(imgHSV, np.array([28,55,115]), np.array([45,255,255]))
"""
cv2.imshow("stick_bin", imgBin)
cv2.waitKey(0)
"""

kernelErosion = np.ones((5,5), np.uint8)
kernelDilation = np.ones((5,5), np.uint8)
imgBin = cv2.erode(imgBin, kernelErosion, iterations=1)
imgBin = cv2.dilate(imgBin, kernelDilation, iterations=1)
imgBin = cv2.GaussianBlur(imgBin, (9,9), 0)
"""
cv2.imshow("stick_bin", imgBin)
cv2.waitKey(0)
"""

_, contours, hierachy = cv2.findContours(imgBin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


"""
print contours[0][0]
cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow("stick with contours", img)
cv2.waitKey(0)
"""

# straight bounding rect

[x,y,w,h] = cv2.boundingRect(contours[0])
print [x,y,w,h]
cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),2)
""" 
# rotate rect
rect =cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
print box
cv2.drawContours(img, [box], 0, (0,0,255),2)
"""

"""
cv2.imshow("stick with contours", img)
cv2.waitKey(0)
"""

