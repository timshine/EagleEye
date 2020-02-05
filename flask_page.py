"""This program will open a flask page to display the live stream
video. It will also offer button for user control"""

# Import necessary packages to open flask page
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# Thread safe exchanges between output frames (multiple browsers)
outputFrame = None
lock = threading.Lock()

# Begin flask object
app = Flask(__name__)


@app.route("/video_feed")               # At the endpoint /video_feed
def webpage():
    # Return the template
    return render_template("webpage.html")

def screen_words():
    return 'Home Screen'

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)