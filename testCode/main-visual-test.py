## ---------------------------------------------------------------------
# author: Meringue
# date: 1/15/2018
# description: some test codes for Nao golf visual part.
## ---------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import time
import os
import sys
sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
sys.path.append("./")

from visualTask import *

from naoqi import ALProxy
import vision_definitions as vd

IP = "192.168.1.101"

visualBasis = VisualBasis(IP,cameraId=0, resolution=vd.kVGA)
ballDetect = BallDetect(IP, resolution=vd.kVGA)
stickDetect = StickDetect(IP, cameraId=0, resolution=vd.kVGA)
objectDetect = ObjectDetection(IP, cameraId=0, resolution=vd.kVGA)



##test code


"""
visualBasis.showFrame()
visualBasis.updateFrame()
visualBasis.showFrame()
visualBasis.printFrameData()
"""


"""
frameArray = visualBasis.getFrameArray()
print "current frame array = ", frameArray
visualBasis.updateFrame()
frameArray = visualBasis.getFrameArray()
print "current frame array = ", frameArray
"""

"""
while 1:
	time1 = time.time()
	ballDetect.updateBallData()
	time2 = time.time()
	print "updare data time = ", time2-time1
	ballDetect.showBallPosition(showTime = 20)
"""


"""
while 1:
	stickDetect.updateStickData()
	stickDetect.showStickPosition(showTime = 10)
"""


"""
print "start collecting..."
for i in range(1000):
	imgName = "stick_" + str(i+127) + ".jpg"
	imgDir = os.path.join("stick_images", imgName)
	visualBasis.updateFrame()
	visualBasis.showFrame(timeMs=1000)
	visualBasis.saveFrame(imgDir)
	print "saved in ", imgDir
	time.sleep(5)
"""


print("stick detection...")
classes_name = ["stick"]
#classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

modelFile = "/home/meringue/Documents/stick_detection_tensorflow/tensorflow-yolo-python2.7/models/train20180308/model.ckpt-95000"
#modelFile = "/home/meringue/Documents/stick_detection_tensorflow/tensorflow-yolo-python2.7/models/pretrain/yolo_tiny.ckpt"

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
object_predicts = objectDetect.predict_single_object(image)
sess = tf.Session()
saver = tf.train.Saver(objectDetect._net.trainable_collection)

#saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
saver.restore(sess, modelFile)

while 1:
	objectDetect.updateFrame()
	frame = objectDetect.getFrameArray()
	height_ratio = 1.0*frame.shape[0]/448
	width_ratio = 1.0*frame.shape[1]/448
	resized_img = cv2.resize(frame, (448, 448))
	np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
	np_img = np_img.astype(np.float32)
	np_img = np_img / 255.0 * 2 - 1
	np_img = np.reshape(np_img, (1, 448, 448, 3))
	predicts = sess.run(object_predicts, feed_dict={image: np_img})
	rect, class_num = objectDetect.process_predicts(predicts)
	rect[0] = rect[0]*width_ratio
	rect[1] = rect[1]*height_ratio
	rect[2] = rect[2]*width_ratio
	rect[3] = rect[3]*height_ratio

	objectDetect.showDetectResult(frame, rect, classes_name[class_num])

	if cv2.waitKey(20) & 0xFF==ord("q"):
		break
sess.close()







"""
visualBasis._tts.say("hello world")
"""

"""
visualBasis._motionProxy.wakeUp()
"""

"""
dataList = visualBasis._memoryProxy.getDataList("camera")
print dataList
"""

"""
visualBasis._motionProxy.setStiffnesses("Body", 1.0)
visualBasis._motionProxy.moveInit()
"""

#motionProxy = ALProxy("ALMotion", IP, 9559)
#postureProxy = ALProxy("ALRobotPosture", IP, 9559)

#motionProxy.wakeUp()
#postureProxy.goToPosture("StandInit", 0.5)


#motionProxy.wakeUp()
#motionProxy.goToPosture("StandInit", 0.5)
#motionProxy.moveToward(0.1, 0.1, 0, [["Frequency", 1.0]])
#motionProxy.moveTo(0.3, 0.2, 0)
"""
"""
