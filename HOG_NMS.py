# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# import the necessary packages

from imutils.object_detection import non_max_suppression
from GoPro import gopro_live
from imutils import paths
import timeit
import numpy as np
import argparse
import imutils
import threading
import cv2
import os
import signal
import time
import socket

def get_command_msg(id):
    return "_GPHD_:%u:%u:%d:%1lf\n" % (0, 0, 2, 0)

outputFrame = None
lock = threading.Lock()

time.sleep(2.0)

def read_video_stream():
    global outputFrame, lock
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()

    #need to call the GoPro code to open the udp stream
    gopro_live()
    print("Returned from gopro_live")

    #open webcam video stream
    #capture = cv2.VideoCapture(0)

    #open goPro Stream
    #must be connected to the GoPro Wifi
    capture = cv2.VideoCapture('udp://10.5.5.100:8554?overrun_nonfatal=1&fifo_size=50000000')

    if not capture.isOpened():
        print('VideoCapture not opened')
        exit(-1)

    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        if ret == True:
            frame = imutils.resize(frame, width=min(1080, frame.shape[1]))
            orig = frame.copy()

            # detect people in the image
            (rects, weights) = hog.detectMultiScale(frame, winStride=(12, 12),padding=(14, 14), scale=1.05)

            # draw the original bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            # Good: pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            t0 = timeit.timeit()
            pick = non_max_suppression_fast(rects, 0.75)
            t1 = timeit.timeit()
            #print("Time in non_max_suppression is %f" %(t1-t0))

            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 255, 255), 2)

            #aquire lock, set the output frame
            with lock:
                outputFrame = frame.copy()
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # Display the resulting frame
            # cv2.imshow('Eye Sight (original)',orig)
            # cv2.imshow('Eye Sight',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +	bytearray(encodedImage) + b'\r\n')

    # When everything done, release the capture
    capture.release()
    # finally, close the window
    cv2.destroyAllWindows()
    cv2.waitKey(1)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes thqat were picked using the
	# integer data type
	return boxes[pick].astype("int")

# def main():
    #read_video_stream
    #os.kill(os.getppid(), signal.SIGHUP)  # closes terminal when script ends

# if __name__ == "__main__":
#     main()
