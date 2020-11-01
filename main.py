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
white_trackbars = TrackbarWindow(white_num, color='white')
green_trackbars = TrackbarWindow(1)




while True:
    if drone.frame is not None:
        img = cv2.resize(drone.frame, (width, height))
        imgContour = img.copy()

        data = white_trackbars.getTrackbarValues() # returns: (h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, areaMin)
        areaMin = data[-1]

        imgDil, result = prepareImg(data, img)

        direction, area, crop_xywh = getContours(imgDil, imgContour, width, height, deadZone, areaMin)
        display(imgContour, width, height, deadZone)


        # candidate
        if crop_xywh is not None:
            x, y, w, h = crop_xywh
            candidate_area = img[y:y+h, x:x+w]
            candidate_area = cv2.resize(candidate_area, (320, 240))
        else:
            candidate_area = np.zeros((240, 320, 3), np.uint8)
        candidate_area = cv2.cvtColor(candidate_area, cv2.COLOR_BGR2GRAY)




        cv2.imshow('candidate', candidate_area)

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
