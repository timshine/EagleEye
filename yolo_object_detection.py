# USAGE
# python yolo_object_detection.py --input ../example_videos/janie.mp4 --output ../output_videos/yolo_janie.avi --yolo yolo-coco --display 0
# python yolo_object_detection.py --input ../example_videos/janie.mp4 --output ../output_videos/yolo_janie.avi --yolo yolo-coco --display 0 --use-gpu 1

# import the necessary packages
from imutils.video import FPS
from imutils.video import VideoStream
from GoProStream import gopro_live
from flask import Response, Flask, render_template, request, url_for, flash
import numpy as np
import argparse
import threading
from threading import Thread
import cv2
import os
import time
from queue import Queue
from urllib.request import urlopen
from color_detection import detect_red

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)
time.sleep(2.0)
"""
if args["use_gopro"] == 0:
    #just getting video from webcam
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
elif args["use_gopro"] == 1:
    #using stream from gopro
    print("[INFO] Trying to use stream from GoPro")
    urlopen("http://10.5.5.9/gp/gpControl/execute?p1=gpStream&a1=proto_v2&c1=restart").read()
    time.sleep(2.0)
    vs = VideoStream('udp://10.5.5.100:8554').start()
else:
    #using steam from DJI
    #mike add your code here
    print("[INFO] Trying to use stream from DJI")
"""
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
#if args["use_gpu"]:
#set CUDA as the preferable backend and target
print("[INFO] setting preferable backend and target to CUDA...")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the width and height of the frames in the video file
W = None
H = None

def yolo_stream(camera):
	global vs, outputFrame, lock, W, H, ln, net
	#vs = cv2.VideoCapture('udp://10.5.5.100:8554')
	if(camera == 0):
		print("[INFO] accessing video stream from webcam...")
		vs = VideoStream(src=0).start()
	elif(camera == 1):
		print("[INFO] accessing video stream from GoPro...")
		vs = VideoStream('udp://10.5.5.100:8554').start()
	# load the COCO class labels our YOLO model was trained on

	# loop over frames from the video file stream
	while True:
		frame = vs.read()

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > .5:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, .5, .3)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				if LABELS[classIDs[i]]=="person":
					color = [int(c) for c in COLORS[classIDs[i]]]
					im = frame[y:(y+h),x:(x+w)]
					percent_red = detect_red(im)
					if percent_red > .4:
						cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
						text = "{}: {:.4f}".format("Hostile", percent_red)
						cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
					else:
						cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
						text = "{}: {:.4f}".format("Friendly", percent_red)
						cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
		with lock:
			outputFrame = frame.copy()

def generate():
	global outputFrame, lock

	while True:
		with lock:
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue


		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +	bytearray(encodedImage) + b'\r\n')
