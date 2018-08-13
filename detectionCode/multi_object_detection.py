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

IP = "192.168.1.105"

visualBasis = VisualBasis(IP,cameraId=0, resolution=vd.kVGA)
ballDetect = BallDetect(IP, resolution=vd.kVGA)
stickDetect = StickDetect(IP, cameraId=0, resolution=vd.kVGA)
objectDetect = ObjectDetection(IP, cameraId=0, resolution=vd.kVGA)
multiObjectDetect = MultiObjectDetection(IP, cameraId=0, resolution=vd.kVGA)





print("object detection...")
#classes_name = ["stick"]
classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

#modelFile = "/home/meringue/Documents/object_detection_tensorflow/tensorflow-yolo-python2.7/models/train/model.ckpt-125000"
modelFile = "/home/meringue/Documents/object_detection_tensorflow/tensorflow-yolo-python2.7/models/pretrain/yolo_tiny.ckpt"

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
object_predicts = multiObjectDetect.predict_object(image)

sess = tf.Session()
saver = tf.train.Saver(multiObjectDetect._net.trainable_collection)
#saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
saver.restore(sess, modelFile)

while 1:
	multiObjectDetect.updateFrame()
	frame = multiObjectDetect.getFrameArray()
	resized_img = cv2.resize(frame, (448, 448))
	height_ratio = frame.shape[0]/448
	width_ratio = frame.shape[1]/448
	# convert to rgb image
	np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

	# convert data type used in tf
	np_img = np_img.astype(np.float32)

	# data normalization and reshape to input tensor
	np_img = np_img / 255.0 * 2 - 1
	np_img = np.reshape(np_img, (1, 448, 448, 3))

	print('Procession detection...')
	np_predict = sess.run(object_predicts, feed_dict={image: np_img})


	predicts_dict = multiObjectDetect.process_predicts(resized_img, np_predict)

	print ("predict dict = ", predicts_dict)
	predicts_dict = multiObjectDetect.non_max_suppress(predicts_dict)
	print ("predict dict = ", predicts_dict)

	multiObjectDetect.plot_result(frame, predicts_dict)	
	if cv2.waitKey(20) & 0xFF==ord("q"):
		break
sess.close()







