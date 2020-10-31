import cv2.cv2 as cv2
import numpy as np

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


def getContours(img, imgContour, frameWidth, frameHeight, deadZone):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    deadZoneW = deadZone
    deadZoneH = deadZone - 15

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 255, 0), 2)

            cx = int(x + (w / 2))
            cy = int(y + (h / 2))

            if (cx < int(frameWidth / 2) - deadZoneW):
                cv2.putText(imgContour, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (0, int(frameHeight / 2 - deadZoneH)),
                              (int(frameWidth / 2) - deadZoneW, int(frameHeight / 2) + deadZoneH), (0, 0, 255),
                              cv2.FILLED)
            elif (cx > int(frameWidth / 2) + deadZoneW):
                cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 + deadZoneW), int(frameHeight / 2 - deadZoneH)),
                              (frameWidth, int(frameHeight / 2) + deadZoneH), (0, 0, 255), cv2.FILLED)
            elif (cy < int(frameHeight / 2) - deadZoneH):
                cv2.putText(imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZoneW), 0),
                              (int(frameWidth / 2 + deadZoneW), int(frameHeight / 2) - deadZoneH), (0, 0, 255),
                              cv2.FILLED)
            elif (cy > int(frameHeight / 2) + deadZoneH):
                cv2.putText(imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZoneW), int(frameHeight / 2) + deadZoneH),
                              (int(frameWidth / 2 + deadZoneW), frameHeight), (0, 0, 255), cv2.FILLED)

            cv2.line(imgContour, (int(frameWidth / 2), int(frameHeight / 2)), (cx, cy),
                     (0, 0, 255), 3)


def display(img, frameWidth, frameHeight, deadZone):
    deadZoneW = deadZone
    deadZoneH = deadZone - 15
    cv2.line(img, (int(frameWidth / 2) - deadZoneW, 0), (int(frameWidth / 2) - deadZoneW, frameHeight), (255, 255, 0), 3)
    cv2.line(img, (int(frameWidth / 2) + deadZoneW, 0), (int(frameWidth / 2) + deadZoneW, frameHeight), (255, 255, 0), 3)

    cv2.circle(img, (int(frameWidth / 2), int(frameHeight / 2)), 5, (0, 0, 255), 5)
    cv2.line(img, (0, int(frameHeight / 2) - deadZoneH), (frameWidth, int(frameHeight / 2) - deadZoneH), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZoneH), (frameWidth, int(frameHeight / 2) + deadZoneH), (255, 255, 0), 3)