## ---------------------------------------------------------------------
# author: Meringue
# date: 1/15/2018
# description: visual classes for Nao golf task.
## ---------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
sys.path.append("./")
#os.chdir(os.getcwd())
import cv2
import numpy as np

import vision_definitions as vd
import time

from configureNao import ConfigureNao
from naoqi import ALProxy
sys.path.append("/home/meringue/Documents/python-nao-golf/yoloNet")
from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf




class VisualBasis(ConfigureNao):
    """
    a basic class for visual task.
    """
    
    def __init__(self, IP, cameraId, resolution=vd.kVGA):
        """
        initilization.
        
        Arguments:
        IP -- NAO's IP
        cameraId -- bottom camera (1,default) or top camera (0).
        resolution -- (kVGA, default: 640*480)
        """  
        
        super(VisualBasis, self).__init__(IP)
        self._cameraId = cameraId
        self._resolution = resolution
        
        self._colorSpace = vd.kBGRColorSpace
        self._fps = 20

        self._frameHeight = 0
        self._frameWidth = 0
        self._frameChannels = 0
        self._frameArray = None
        
        self._cameraPitchRange = 47.64/180*np.pi
        self._cameraYawRange = 60.97/180*np.pi
        
        
    def updateFrame(self):
        """
        get a new image from the specified camera and save it in self._frame.
        """

        time2 = time.time()
        if self._cameraProxy.getActiveCamera() == self._cameraId:
            print("current camera has been actived.")
        else:
            self._cameraProxy.setActiveCamera(self._cameraId)
        time3 = time.time()
        videoClient = self._cameraProxy.subscribe("python_client", self._resolution, self._colorSpace, self._fps)
        time4 = time.time()
        frame = self._cameraProxy.getImageRemote(videoClient)
        time5 = time.time()
        self._cameraProxy.unsubscribe(videoClient)

        print("active camera time = %f s." %(time3-time2))
        print("subscribe time = %f s." %(time4-time3))
        print("get imafe romote time = %f s." %(time5-time4))
        
        self._frameWidth = frame[0]
        self._frameHeight = frame[1]
        self._frameChannels = frame[2]
        self._frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1],frame[0],frame[2]])
        
    
    def getFrameArray(self):
		"""
		return current frame
		"""
		if self._frameArray is None:
			return np.array([])
		return self._frameArray
		
		    
    def showFrame(self, timeMs=1000):
        """
        show current frame image.
        """

        if self._frameArray is None:
            print("please get an image from Nao with the method updateFrame()")
        else:
			cv2.imshow("current frame", self._frameArray)
			cv2.waitKey(timeMs)
			
    
    def printFrameData(self):
        """
        print current frame data.
        """
        print("frame height = ", self._frameHeight)
        print("frame width = ", self._frameWidth)
        print("frame channels = ", self._frameChannels)
        print("frame shape = ", self._frameArray.shape)
        
    
    
    def saveFrame(self, framePath):
		"""
		save current frame to specified direction.
		
		Arguments:
		framePath -- image path.
		"""
		
		cv2.imwrite(framePath, self._frameArray)
		print("current frame image has been saved in", framePath)
		
		  
    def setParam(self, paramName=None, paramValue = None):
        pass
    
    
    def setAllParamsToDefault(self):
        pass
        



class BallDetect(VisualBasis):
    """
    derived from VisualBasics, used to detect the ball.
    """
    
    def __init__(self, IP, cameraId=vd.kBottomCamera, resolution=vd.kVGA):
        """
        initialization.
        """
        super(BallDetect, self).__init__(IP, cameraId, resolution)
        self._ballData = {"centerX":0, "centerY":0, "radius":0}
        self._ballPosition= {"disX":0, "disY":0, "angle":0}

    
    def _getChannelAndBlur(self, Hm, color):
        """
        get the specified channel and blur the result.
        
        Arguments:
        Hm -- paramater related to channel split, usual in the range of (5, 8).
        color -- the color channel to split, only supports the color of red, geen and blue.
        
        Return: the specified color channel or None (when the color is not supported).
        """
        channelB = self._frameArray[:,:,0]
        channelG = self._frameArray[:,:,1]
        channelR = self._frameArray[:,:,2]
        
        if color == "red":
            channelB = channelB*0.1*Hm
            channelG = channelG*0.1*Hm
            channelR = channelR - channelB - channelG
            channelR = 3*channelR
            channelR = cv2.GaussianBlur(channelR, (9,9), 1.5)
            channelR[channelR<0] = 0
            channelR[channelR>255] = 255
            return np.uint8(np.round(channelR))
        
        elif color == "blue":
            channelR = channelR*0.1*Hm
            channelG = channelG*0.1*Hm
            channelB = channelB - channelG - channelR
            channelB = 3*channelB            
            channelB = cv2.GaussianBlur(channelB, (9,9), 1.5)
            channelB[channelB<0] = 0
            channelB[channelB>255] = 255
            return np.uint8(np.round(channelB))
        
        elif color == "green":
            channelB = channelB*0.1*Hm
            channelR= channelR*0.1*Hm
            channelG = channelG - channelB - channelR
            channelG = 3*channelG
            channelG = cv2.GaussianBlur(channelG, (9,9), 1.5)
            channelG[channelG<0] = 0
            channelG[channelG>255] = 255
            return np.uint8(np.round(channelG))
        
        else:
            print("can not recognize the color!")
            print("supported color:red, green and blue.")
            return None
    
    
    def _findCircles(self, img, minDist, minRadius, maxRadius):
        """
        detect circles from an image.
        
        Arguments:
        img -- image to be detected.
        minDist -- minimum distance between the centers of the detected circles.
        minRadius -- minimum circle radius.
        maxRadius -- maximum circle radius.
        
        Return: an uint16 numpy array shaped circleNum*3 if circleNum>0, ([[circleX, circleY,radius]])
                else return None.
        """
        circles = cv2.HoughCircles(np.uint8(img), cv2.HOUGH_GRADIENT, 1, minDist, 
                                   param1=150, param2=15, minRadius=minRadius, maxRadius=maxRadius)
        
        # print "detected circle = ", circles
        
        if circles is None:
            return np.uint16([])
        else:
            return np.uint16(np.around(circles[0, ]))
    
    
    def _selectCircle(self, circles):
        """
        select one circle in list type from all circles detected. if no circle is selected, return None.
        """
        
        if len(circles) == 0 :
            return circles
        
        if circles.shape[0] == 1:
            centerX = circles[0][0]
            centerY = circles[0][1]
            radius = circles[0][2]
            initX = centerX - 2*radius
            initY = centerY - 2*radius
            if initX<0 or initY<0 or (initX+4*radius)>self._frameWidth or (initY+4*radius)>self._frameHeight or radius<1:
                return circles
            
        channelB = self._frameArray[:,:,0]
        channelG = self._frameArray[:,:,1]
        channelR = self._frameArray[:,:,2]
        
        rRatioMin = 1.0; circleSelected = np.uint16([])
        for circle in circles:
            centerX = circle[0]
            centerY = circle[1]
            radius = circle[2]
            initX = centerX - 2*radius
            initY = centerY - 2*radius
            
            if initX<0 or initY<0 or (initX+4*radius)>self._frameWidth or (initY+4*radius)>self._frameHeight or radius<1:
                continue
                
            rectBallArea = self._frameArray[initY:initY+4*radius+1, initX:initX+4*radius+1,:]
            bFlat = np.float16(rectBallArea[:,:,0].flatten())
            gFlat = np.float16(rectBallArea[:,:,1].flatten())
            rFlat = np.float16(rectBallArea[:,:,2].flatten())
            
            rScore1 = np.uint8(rFlat>1.0*gFlat)
            rScore2 = np.uint8(rFlat>1.0*bFlat)
            rScore = float(np.sum(rScore1*rScore2))
            
            gScore = float(np.sum(np.uint8(gFlat>1.0*rFlat)))
            
            rRatio = rScore/len(rFlat)
            gRatio = gScore/len(gFlat) 
            
            print("red ratio = ", rRatio)
            print("green ratio = ", gRatio)
            
            if rRatio>=0.12 and gRatio>=0.1 and rRatio<rRatioMin:
                circleSelected = circle
                
        return circleSelected
    
    
    def _updateBallPosition(self, standState):
        """
        compute and update the ball position with the ball data in frame.
        """
        
        bottomCameraDirection = {"standInit":49.2, "standUp":39.7} 
        try:
            cameraDirection = bottomCameraDirection[standState]
        except KeyError:
            print("Error! unknown standState, please check the value of stand state!")
        else:
            if self._ballData["radius"] == 0:
                self._ballPosition= {"disX":0, "disY":0, "angle":0}
            else:
                centerX = self._ballData["centerX"]
                centerY = self._ballData["centerY"]
                radius = self._ballData["radius"]
                pass #unfinished
            
                               
    def updateBallData(self, standState="standInit", Hm=6, color="red"):
        """
        update the ball data with the frame get from the bottom camera.
        
        Arguments:
        standState -- ("standInit", default), "standInit" or "standUp".
        Hm -- (6, default) param related with color split.
        color -- ("red", default) the color of ball to be detected.
        
        Return: a dict with ball data. for example: {"centerX":0, "centerY":0, "radius":0}.
                if no ball be detected, all key values in the dict are 0.
        """
        
        self.updateFrame()
        minDist = int(self._frameHeight/30.0)
        minRadius = 1
        maxRadius = int(self._frameHeight/10.0)
        grayFrame = self._getChannelAndBlur(Hm, color)
        cv2.imshow("bin frame", grayFrame)
        cv2.waitKey(20)
        circles = self._findCircles(grayFrame, minDist, minRadius, maxRadius)
        circle = self._selectCircle(circles)
        
        if len(circle) == 0:
            self._ballData = {"centerX":0, "centerY":0, "radius":0}
            self._ballPosition= {"disX":0, "disY":0, "angle":0}
        else:    
            self._ballData = {"centerX":circle[0], "centerY":circle[1], "radius":circle[2]}
            self._updateBallPosition(standState=standState);
          
        
    def getBallPostion(self):
        """
        get ball position.
        
        Return: distance in x axis, distance in y axis and direction related to Nao.
        """
        return self._ballPosition["disX"], self._ballPosition["disY"], self._ballPosition["angle"] 
 
        
    def showBallPosition(self, showTime = 1000):        
        """
        show ball data in the current frame.
        """
        
        if self._ballData["radius"] == 0:
            print("no ball found.")
            cv2.imshow("ball position", self._frameArray)
            cv2.waitKey(showTime)
        else:
            print("ball postion = ", (self._ballPosition["disX"], self._ballPosition["disY"]))
            print("ball direction = ", self._ballPosition["angle"])
            frameArray = self._frameArray
            cv2.circle(frameArray, (self._ballData["centerX"],self._ballData["centerY"]),
                       self._ballData["radius"], (250,150,150),2)
            cv2.circle(frameArray, (self._ballData["centerX"],self._ballData["centerY"]),
                       2, (50,250,50), 3)
            cv2.imshow("ball position", frameArray)
            cv2.waitKey(showTime)


class StickDetect(VisualBasis):
	"""
	derived from VisualBasics, used to detect the stict.
	"""
	
	def __init__(self, IP, cameraId=vd.kTopCamera, resolution=vd.kVGA):
		super(StickDetect, self).__init__(IP, cameraId, resolution)
		self._boundRect = []
		self._cropKeep = 1
		self._stickAngle = None # rad
        

	def _preprocess(self, minHSV, maxHSV, cropKeep, morphology):
		"""
		preprocess the current frame for stick detection.
		(binalization, crop etc.)
		
		Arguments:
		minHSV -- the lower limit for binalization.
		maxHSV -- the upper limit for binalization.
		cropKeep --  crop ratio (>=0.5).
		morphology -- erosion and dilation.
		
		Return: preprocessed image for stick detection.
		"""
		
		if self._frameArray is None: # try  ,except, else
			print("please update the frame data!")
			return
			
		self._cropKeep = cropKeep
			
		frameArray = self._frameArray
		height = self._frameHeight
		width = self._frameWidth
		
		frameArray = frameArray[int((1-cropKeep)*height):,:]
					
		frameHSV = cv2.cvtColor(frameArray, cv2.COLOR_BGR2HSV)
		frameBin = cv2.inRange(frameHSV, minHSV, maxHSV)
		
		kernelErosion = np.ones((5,5), np.uint8)
		kernelDilation = np.ones((5,5), np.uint8) 
		frameBin = cv2.erode(frameBin, kernelErosion, iterations=1)
		frameBin = cv2.dilate(frameBin, kernelDilation, iterations=1)
		frameBin = cv2.GaussianBlur(frameBin, (9,9), 0)
		
		cv2.imshow("stick bin", frameBin)
		cv2.waitKey(20)
		
		return frameBin
		
		
	def _findStick(self, frameBin, minPerimeter, minArea):
		"""
		find the yellow stick in the preprocessed frame.
		
		Arguments:
		frameBin -- preprocessed frame.
		minPerimeter minimum perimeter of detected stick.
		minArea -- minimum area of detected stick.
		
		Return: detected stick marked with rectangle or [].
		"""
		
		rects = []
		_, contours, _ = cv2.findContours(frameBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		
		if len(contours) == 0:
			return rects

		for contour in contours:
			perimeter = cv2.arcLength(contour, True)
			area = cv2.contourArea(contour)
			if perimeter>minPerimeter and area>minArea:
				x,y,w,h = cv2.boundingRect(contour)
				rects.append([x,y,w,h])
				
		if len(rects) == 0:
			return rects
				
		rects = [rect for rect in rects if (1.0*rect[3]/rect[2])>0.8]
		
		if len(rects) == 0:
			return rects
			
		rects = np.array(rects)
		print(rects)
		rect = rects[np.argmax(1.0*(rects[:,-1])/rects[:,-2]),]
		rect[1] += int(self._frameHeight *(1-self._cropKeep))
		return rect
		
		
	def updateStickData(self, minHSV=np.array([27,55,115]), maxHSV=np.array([45,255,255]), cropKeep=1, morphology=True):
		"""
		update the yellow stick data from the specified camera.
		
		Arguments:
		minHSV -- the lower limit for binalization.
		maxHSV -- the upper limit for binalization.
		cropKeep --  crop ratio (>=0.5).
		morphology -- (True, default), erosion and dilation.
		"""

		self.updateFrame()
		minPerimeter = self._frameHeight/8.0
		minArea = self._frameHeight*self._frameWidth/1000.0
		
		frameBin = self._preprocess(minHSV, maxHSV, cropKeep, morphology)
		rect = self._findStick(frameBin, minPerimeter, minArea)
		
		if rect == []:
			self._boundRect = []
			self._stickAngle = None
		else:
			self._boundRect = rect
			centerX = rect[0]+rect[2]/2
			width = self._frameWidth *1.0
			self._stickAngle = (width/2-centerX)/width*self._cameraYawRange
			cameraAngle = 0 # get the current camera  yaw angle (unfinished)
			self._stickAngle += cameraAngle
			
		
	def showStickPosition(self, showTime=1000):
		"""
		show the stick  position in the current frame.
		"""
		if self._boundRect == []:
			print("no stick detected.")
			cv2.imshow("stick position", self._frameArray)
		else:
			[x,y,w,h] = self._boundRect
			frame = self._frameArray.copy()
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
			cv2.imshow("stick position", frame)
			cv2.waitKey(showTime)



class ObjectDetection(VisualBasis):

    def __init__(self, IP, cameraId=vd.kTopCamera, resolution=vd.kVGA):
        super(ObjectDetection, self).__init__(IP, cameraId, resolution)
        self._boundRect = []
        self._cropKeep = 1
        self._stickAngle = None # rad
        #self._classes_name = ["stick"]
        self._common_params = {'image_size': 448, 'num_classes': 1, 
                'batch_size':1}
        self._net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}
        self._net = YoloTinyNet(self._common_params, self._net_params, test=True)
        #self._modelFile = "/home/meringue/Documents/python-nao-golf/yoloNet/models/train/model.ckpt-95000"
        #self._objectRect = [0, 0, 0, 0]
        self._objectName = None

    def predict_single_object(self, image):
        predicts = self._net.inference(image)
        return predicts


    def process_predicts(self, predicts):
        p_classes = predicts[0, :, :, 0:1]
        C = predicts[0, :, :, 1:3]
        coordinate = predicts[0, :, :, 3:]

        p_classes = np.reshape(p_classes, (7, 7, 1, 1))
        C = np.reshape(C, (7, 7, 2, 1))

        P = C * p_classes

        index = np.argmax(P)
        print("confidence = ", np.max(P))
        index = np.unravel_index(index, P.shape)

        class_num = index[3]
        coordinate = np.reshape(coordinate, (7, 7, 2, 4))
        max_coordinate = coordinate[index[0], index[1], index[2], :]
        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[1] + xcenter) * (448/7.0)
        ycenter = (index[0] + ycenter) * (448/7.0)

        w = w * 448
        h = h * 448

        xmin = xcenter - w/2.0
        ymin = ycenter - h/2.0

        xmax = xmin + w
        ymax = ymin + h

        return [xmin, ymin, xmax, ymax], class_num

    def showDetectResult(self, frame, rect, object_name):
        object_min_xy = (int(rect[0]), int(rect[1]))
        object_max_xy = (int(rect[2]), int(rect[3]))
        cv2.rectangle(frame, object_min_xy, object_max_xy, (0, 0, 255))
        cv2.putText(frame, object_name, object_min_xy, 2, 2, (0, 0, 255))
        cv2.imshow("detect result", frame)
        #cv2.waitKey(10)




				
