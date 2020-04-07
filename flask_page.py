"""This program will open a flask page to display the live stream
video. It will also offer button for user control"""

# Import necessary packages to open flask page
import os
from HOG_NMS import read_video_stream
from flask import Response, Flask, render_template
from threading import Thread
import argparse
import datetime
import imutils
import time
import cv2

# Begin flask object
app = Flask(__name__)


@app.route("/")               
def webpage():
    # Return the template
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
    return Response(read_video_stream(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__=='__main__':
    # start a thread that will perform motion detection
    # t1 = Thread(target = read_video_stream)
    # t1.start()
    # start flask app
    app.run(host='0.0.0.0', debug=True, threaded=True)