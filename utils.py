import cv2.cv2 as cv2
import numpy as np
from typing import Optional


def empty(a):
    pass


# class used to create trackbar window and get their parameters
class TrackbarWindow():
    def __init__(self, num, color=None):
        if color == 'white':  #predefined values for mapping whites only
            init_h_min, init_h_max, init_s_min, init_s_max, init_v_min, init_v_max = 0, 179, 0, 59, 220, 255
            init_thr1, init_thr2, init_area = 166, 171, 2000
        elif color == 'green':
            init_h_min, init_h_max, init_s_min, init_s_max, init_v_min, init_v_max = 75, 90, 30, 170, 40, 245
            init_thr1, init_thr2, init_area = 0, 255, 0
        else:
            init_h_min, init_h_max, init_s_min, init_s_max, init_v_min, init_v_max = 0, 179, 0, 255, 0, 255
            init_thr1, init_thr2, init_area = 0, 255, 0

        cv2.namedWindow(f"HSV_{num}")
        cv2.resizeWindow(f"HSV_{num}", 640, 240)
        cv2.createTrackbar(f"HUE Min_{num}", f"HSV_{num}", init_h_min, 179, empty)
        cv2.createTrackbar(f"HUE Max_{num}", f"HSV_{num}", init_h_max, 179, empty)
        cv2.createTrackbar(f"SAT Min_{num}", f"HSV_{num}", init_s_min, 255, empty)
        cv2.createTrackbar(f"SAT Max_{num}", f"HSV_{num}", init_s_max, 255, empty)
        cv2.createTrackbar(f"VALUE Min_{num}", f"HSV_{num}", init_v_min, 255, empty)
        cv2.createTrackbar(f"VALUE Max_{num}", f"HSV_{num}", init_v_max, 255, empty)

        cv2.namedWindow(f"Parameters_{num}")
        cv2.resizeWindow(f"Parameters_{num}", 640, 240)
        cv2.createTrackbar(f"Threshold1_{num}", f"Parameters_{num}", init_thr1, 255, empty)
        cv2.createTrackbar(f"Threshold2_{num}", f"Parameters_{num}", init_thr2, 255, empty)
        cv2.createTrackbar(f"Area_{num}", f"Parameters_{num}", init_area, 30000, empty)
        self.num = num

    def getTrackbarValues(self):
        h_min = cv2.getTrackbarPos(f"HUE Min_{self.num}", f"HSV_{self.num}")
        h_max = cv2.getTrackbarPos(f"HUE Max_{self.num}", f"HSV_{self.num}")
        s_min = cv2.getTrackbarPos(f"SAT Min_{self.num}", f"HSV_{self.num}")
        s_max = cv2.getTrackbarPos(f"SAT Max_{self.num}", f"HSV_{self.num}")
        v_min = cv2.getTrackbarPos(f"VALUE Min_{self.num}", f"HSV_{self.num}")
        v_max = cv2.getTrackbarPos(f"VALUE Max_{self.num}", f"HSV_{self.num}")

        threshold1 = cv2.getTrackbarPos(f"Threshold1_{self.num}", f"Parameters_{self.num}")
        threshold2 = cv2.getTrackbarPos(f"Threshold2_{self.num}", f"Parameters_{self.num}")
        areaMin = cv2.getTrackbarPos(f"Area_{self.num}", f"Parameters_{self.num}")

        return h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, areaMin


# code by Murtaza's Workshop (Youtube)
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def prepareImg(data, img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, areaMin = data

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
    return imgDil, result

def getCandidateContours(img, imgContour, areaMin):
    direction = 0
    area = 0
    crop_xywh = None

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            crop_xywh = x, y, w, h

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

    if area is not 0 :
        return 1
    else:
        return 0

def getContours(raw_img, img, imgContour, frameWidth, frameHeight, deadZone, areaMin, green_trackbars):
    direction = 0
    area = 0
    crop_xywh = None

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    deadZoneW = deadZone
    deadZoneH = deadZone - 15

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            crop_xywh = x, y, w, h

            if crop_xywh is not None:
                x, y, w, h = crop_xywh
                candidate_img = raw_img[y:y + h, x:x + w]
                candidate_img = cv2.resize(candidate_img, (320, 240))
            else:
                candidate_img = np.zeros((240, 320, 3), np.uint8)

            candContour = candidate_img.copy()
            data_green = green_trackbars.getTrackbarValues()
            areaMin_green = data_green[-1]

            candDil, candRes = prepareImg(data_green, candidate_img)
            correctCandidate = getCandidateContours(candDil, candContour, areaMin_green)

            stack_cand = stackImages(1, ([candidate_img, candRes], [candDil, candContour]))
            cv2.imshow('candidate', stack_cand)

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            if correctCandidate:
                # cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                #             (0, 255, 0), 2)
                # cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                #             (0, 255, 0), 2)
                # print(area)
                # cv2.putText(imgContour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,
                #             0.7,
                #             (0, 255, 0), 2)

                cx = int(x + (w / 2))
                cy = int(y + (h / 2))

                if (cx < int(frameWidth / 2) - deadZoneW):
                    cv2.putText(imgContour, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    cv2.rectangle(imgContour, (0, int(frameHeight / 2 - deadZoneH)),
                                  (int(frameWidth / 2) - deadZoneW, int(frameHeight / 2) + deadZoneH), (0, 0, 255),
                                  cv2.FILLED)
                    direction = 1
                elif (cx > int(frameWidth / 2) + deadZoneW):
                    cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    cv2.rectangle(imgContour, (int(frameWidth / 2 + deadZoneW), int(frameHeight / 2 - deadZoneH)),
                                  (frameWidth, int(frameHeight / 2) + deadZoneH), (0, 0, 255), cv2.FILLED)
                    direction = 2
                elif (cy < int(frameHeight / 2) - deadZoneH):
                    cv2.putText(imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZoneW), 0),
                                  (int(frameWidth / 2 + deadZoneW), int(frameHeight / 2) - deadZoneH), (0, 0, 255),
                                  cv2.FILLED)
                    direction = 3
                elif (cy > int(frameHeight / 2) + deadZoneH):
                    cv2.putText(imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZoneW), int(frameHeight / 2) + deadZoneH),
                                  (int(frameWidth / 2 + deadZoneW), frameHeight), (0, 0, 255), cv2.FILLED)
                    direction = 4
                else:
                    direction = 0

                cv2.line(imgContour, (int(frameWidth / 2), int(frameHeight / 2)), (cx, cy),
                         (0, 0, 255), 3)

                return direction, area, crop_xywh
            else:
                continue
    return direction, area, crop_xywh


def display(img, frameWidth, frameHeight, deadZone):
    deadZoneW = deadZone
    deadZoneH = deadZone - 15
    cv2.line(img, (int(frameWidth / 2) - deadZoneW, 0), (int(frameWidth / 2) - deadZoneW, frameHeight), (255, 255, 0),
             3)
    cv2.line(img, (int(frameWidth / 2) + deadZoneW, 0), (int(frameWidth / 2) + deadZoneW, frameHeight), (255, 255, 0),
             3)

    cv2.circle(img, (int(frameWidth / 2), int(frameHeight / 2)), 5, (0, 0, 255), 5)
    cv2.line(img, (0, int(frameHeight / 2) - deadZoneH), (frameWidth, int(frameHeight / 2) - deadZoneH), (255, 255, 0),
             3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZoneH), (frameWidth, int(frameHeight / 2) + deadZoneH), (255, 255, 0),
             3)
