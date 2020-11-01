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

if fly:
    drone.takeoff()

# create trackbars
white_num = 0
white_trackbars = TrackbarWindow(white_num, color='white')
green_trackbars = TrackbarWindow(1)







drone.streamon()

while True:
    if drone.frame is not None:
        img = cv2.resize(drone.frame, (width, height))
        imgContour = img.copy()
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, areaMin = white_trackbars.getTrackbarValues()
        print(h_min, h_max, s_min, s_max, v_min, v_max)


        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        imgBlur = cv2.GaussianBlur(result, (5, 5), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
        direction, area = getContours(imgDil, imgContour, width, height, deadZone, areaMin)
        display(imgContour, width, height, deadZone)

        stack = stackImages(1, ([img, result], [imgDil, imgContour]))

        cv2.imshow('Horizontal Stacking', stack)

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

        print(fb)

        drone.rc_control(lr, fb, ud, yaw)

        # cv2.imshow('Cap', img)
        # cv2.imshow('Cap2', imgHsv)





        # Video Stream is closed if escape key is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
drone.land()
