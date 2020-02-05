"""This program will open a flask page to display the live stream
video. It will also offer button for user control"""

# Import necessary packages to open flask page
import os
from HOG_NMS import generate
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# Begin flask object
app = Flask(__name__)


@app.route("/video_feed")               # At the endpoint /video_feed
def webpage():
    # Return the template
    return render_template("webpage.html")

app.run(debug=True)

@app.route("/video")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

def screen_words():
    return 'Home Screen'

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)