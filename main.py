from easytello_fix import PokeTello
import numpy as np
import cv2
from utils import *

width = 320
height = 240
deadZone = 70

fly = 0

drone = PokeTello()
print(drone.get_battery())
drone.streamon()

if fly:
    drone.takeoff()

# create trackbars
white_num = 0
green_num = 1
white_trackbars = TrackbarWindow(white_num, color='white')
green_trackbars = TrackbarWindow(green_num, color='green')

while True:
    if drone.frame is not None:
        img = cv2.resize(drone.frame, (width, height))
        imgContour = img.copy()

        data_white = white_trackbars.getTrackbarValues() # returns: (h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, areaMin)
        areaMin_white = data_white[-1]

        imgDil, result = prepareImg(data_white, img)

        direction, area, crop_xywh = getContours(img, imgDil, imgContour, width, height, deadZone, areaMin_white, green_trackbars)
        display(imgContour, width, height, deadZone)


        # candidate processing

        stack = stackImages(1, ([img, result], [imgDil, imgContour]))
        cv2.imshow('Horizontal Stacking', stack)



        # steering
        lr, fb, ud, yaw = 0, 0, 0, 0
        if direction == 1:
            yaw = -40
        elif direction == 2:
            yaw = 40
        elif direction == 3:
            ud = 40
        elif direction == 4:
            ud = -40
        else:
            ud, yaw = 0, 0

        if area < 4000 and area != 0:
            fb = 10
        elif area > 5000:
            fb = -10
        else:
            fb = 0

        # print(fb)

        drone.rc_control(lr, fb, ud, yaw)

        # cv2.imshow('Cap', img)
        # cv2.imshow('Cap2', imgHsv)





        # Video Stream is closed if escape key is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
drone.land()
