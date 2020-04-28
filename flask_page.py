"""This program will open a flask page to display the live stream
video. It will also offer button for user control"""

# Import necessary packages to open flask page
import os
from HOG_NMS import read_video_stream
from yolo_object_detection import yolo_stream
from flask import Response, Flask, render_template, request, url_for, flash
from urllib.request import urlopen
import threading
from threading import Thread
import argparse
import datetime
import imutils
import time
#import cv2
import sys_information

# Begin flask object
app = Flask(__name__)

outputFrame = None
lock = threading.Lock()

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

@app.route("/")
def original():
    # Return the template
    return render_template("index.html")

# Used to send video stream from NMS to live_stream_video.html
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
    return Response(yolo_stream(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__=='__main__':
    # start a thread that will perform motion detection
    # t1 = Thread(target = read_video_stream)
    # t1.start()
    #t = Thread(target=yolo_stream, args=(outputFrame,lock))
    #t.daemon = True
    #t.start()
    # start flask app
    #app.run(host='0.0.0.0', port='8000', debug=True, threaded=True, use_reloader=False
    app.run(host='0.0.0.0', port='8000', debug=True, threaded=True)
