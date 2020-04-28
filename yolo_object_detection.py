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

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--use-gopro", type=int, default=0,
	help="boolean indicating if GoPro should be used")
ap.add_argument("-u", "--use-gpu", type=bool, default=0,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

"""
if args["use_gopro"] == 0:
    #just getting video from webcam
    vs = VideoStream(src=0).start()
    #vs = cv2.VideoCaputre(0)
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

@app.route("/")
def original():
    # Return the template
    return render_template("index.html")

def yolo_stream():
	#global vs, outputFrame, lock
	global outputFrame, lock
	cv2.startWindowThread()
	vs = VideoStream(src=0).start()
	# urlopen("http://10.5.5.9/gp/gpControl/execute?p1=gpStream&a1=proto_v2&c1=restart").read()
	# time.sleep(2.0)
	# vs = VideoStream('udp://10.5.5.100:8554').start()
	# load the COCO class labels our YOLO model was trained on
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

	print("[INFO] accessing video stream...")

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
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

					text = "{}: {:.4f}".format(LABELS[classIDs[i]],
						confidences[i])
					cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		with lock:
			outputFrame = frame.copy()
			if outputFrame is None:
				continue
			
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			if not flag:
				continue

	
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +	bytearray(encodedImage) + b'\r\n')
	vs.stop()

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

# Used to send video stream from NMS to live_stream_video.html
@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(yolo_stream(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/live_stream_video.html", methods=['POST', 'GET'])
def buttonClickOnVideo():
    # Return the template
    if request.method == 'POST':
        if request.form['videoButton'] == 'target':
            output = "Mis-identified"
        elif request.form['videoButton'] == 'home':
            output = "Return Home"
        print(output)
    return render_template("live_stream_video.html")

@app.route("/about.html")
def about():
    # Return the template
    return render_template("about.html")

@app.route("/index.html")
def home():
    # Return the template
    return render_template("index.html")

@app.route("/l3harris.html")
def l3harris():
    # Return the template
    return render_template("l3harris.html")

@app.route("/sys_stats.html")
def sys_stats():
    # Return the template
    sys_info = sys_information.get_sys_info()
    cpu_info = sys_information.get_cpu_cores()
    boot_time = sys_information.get_boot_time()
    memory_info = sys_information.get_memory_info()
    return render_template("sys_stats.html", sys_info=sys_info, cpu_info=cpu_info, boot_time = boot_time, memory_info=memory_info)

@app.route("/sys_stats.html", methods=['POST'])
def update_stats():
    #updating stats
    sys_info = sys_information.get_sys_info()
    cpu_info = sys_information.get_cpu_cores()
    boot_time = sys_information.get_boot_time()
    memory_info = sys_information.get_memory_info()
    return render_template('sys_stats.html', sys_info=sys_info, cpu_info=cpu_info, boot_time=boot_time, memory_info=memory_info)

if __name__=='__main__':
    # start a thread that will perform motion detection
    #t = Thread(target=yolo_stream)
    #t.daemon = True
    #t.start()

    # start flask app
    app.run(host='0.0.0.0', port='8000', debug=True, threaded=True, use_reloader=False)


