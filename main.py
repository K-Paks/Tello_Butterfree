from easytello_fix import PokeTello
import numpy as np
import cv2
from utils import *

width = 320
height = 240
fly = 1


drone = PokeTello()
print(drone.get_battery())

if fly:
    drone.takeoff()


def empty(a):
    pass


deadZone = 70

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 59, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 220, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 166, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 171, 255, empty)
cv2.createTrackbar("Area", "Parameters", 2000, 30000, empty)


drone.streamon()

while True:
    if drone.frame is not None:
        img = cv2.resize(drone.frame, (width, height))
        imgContour = img.copy()
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("HUE Min","HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "HSV")
        v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
        v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        imgBlur = cv2.GaussianBlur(result, (5, 5), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
        direction, area = getContours(imgDil, imgContour, width, height, deadZone)
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
