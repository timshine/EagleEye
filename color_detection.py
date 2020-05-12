import cv2
import numpy as np

def detect_red(image):
    """Detects the percentage red within a frame. Takes in numpy array or image and the percentage
    of red required to classify as enemy (percent given as a decimal)"""
    # Read in image
    #img = cv2.imread(image)                 #Used if this is an image
    img = image

    if img is not None:
        size = img.size
      	# converting from BGR to HSV color space
        try:
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        except:
            return 0

        # Range for lower red
        lower_red = np.array([0,120,70])
        upper_red = np.array([15,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        number_red_pixels1 = cv2.countNonZero(mask1)

        # Range for upper range
        lower_red = np.array([165,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
        number_red_pixels2 = cv2.countNonZero(mask2)

        frac_red = np.divide(float(number_red_pixels1 + number_red_pixels2), int(size))
        percent_red = round(3 * frac_red, 5)
    else:
        percent_red = 0

    return percent_red



