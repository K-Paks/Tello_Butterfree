import cv2.cv2 as cv2
import numpy as np
from utils import *


def empty(a):
    pass


# class used to create trackbar window and get their parameters
class TrackbarWindow:
    """Class used to create trackbar window and store it's values."""
    def __init__(self, num, color=None):
        if color == 'white':  # predefined values for mapping whites only
            init_h_min, init_h_max, init_s_min, init_s_max, init_v_min, init_v_max = 0, 179, 0, 59, 220, 255
            init_thr1, init_thr2, init_area = 166, 171, 2000
        elif color == 'green':
            init_h_min, init_h_max, init_s_min, init_s_max, init_v_min, init_v_max = 55, 90, 30, 170, 40, 245
            init_thr1, init_thr2, init_area = 0, 255, 2500
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

    def get_trackbar_values(self):
        """Method used to getting the trackbars' values."""
        h_min = cv2.getTrackbarPos(f"HUE Min_{self.num}", f"HSV_{self.num}")
        h_max = cv2.getTrackbarPos(f"HUE Max_{self.num}", f"HSV_{self.num}")
        s_min = cv2.getTrackbarPos(f"SAT Min_{self.num}", f"HSV_{self.num}")
        s_max = cv2.getTrackbarPos(f"SAT Max_{self.num}", f"HSV_{self.num}")
        v_min = cv2.getTrackbarPos(f"VALUE Min_{self.num}", f"HSV_{self.num}")
        v_max = cv2.getTrackbarPos(f"VALUE Max_{self.num}", f"HSV_{self.num}")

        threshold1 = cv2.getTrackbarPos(f"Threshold1_{self.num}", f"Parameters_{self.num}")
        threshold2 = cv2.getTrackbarPos(f"Threshold2_{self.num}", f"Parameters_{self.num}")
        area_min = cv2.getTrackbarPos(f"Area_{self.num}", f"Parameters_{self.num}")

        return h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, area_min


def get_candidate_contours(img, candidate_w_contours, area_min):
    """If sufficient candidate is found, returns `1` and draws contours and boundingRect around it.

    Parameters:
    img (np.ndarray): Image preprocessed by `prepare_img` function.
    candidate_w_contours (np.ndarray): Copy of the original image, on which the contours are going to be applied
    area_min (int): Minimal area of an object to become a candidate

    Returns:
    int: 1 if candidate found, else 0.
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_min:
            cv2.drawContours(candidate_w_contours, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            cv2.rectangle(candidate_w_contours, (x, y), (x + w, y + h), (0, 255, 0), 5)
            return 1
    return 0


def create_candidates(crop_xywh, raw_img, cnt):
    """Function that, if a object is found, returns a candidate image of it.

    Parameters:
    crop_xywh (np.ndarray): an array of bounding box values [x,  y, width, height].
    raw_img (np.ndarray): original drone image.
    cnt (np.ndarray): contours for a given object.

    Returns:
    candidate_img (np.ndarray): the original image cropped into the found object bounding box's size.
    candidate_cropped (np.ndarray): a cropped version of the candidate image.
    """
    if crop_xywh is not None:
        x, y, w, h = crop_xywh
        candidate_img = raw_img[y:y + h, x:x + w]
        candidate_img = cv2.resize(candidate_img, (320, 240))

        mask = np.ones(raw_img.shape[:2])
        mask = cv2.drawContours(mask, [cnt], -1, 0, cv2.FILLED)
        candidate_cropped = raw_img.copy()
        candidate_cropped[mask.astype(np.bool), :] = 0

        candidate_cropped = candidate_cropped[y:y + h, x:x + w]
        candidate_cropped = cv2.resize(candidate_cropped, (320, 240))
    else:
        candidate_img = np.zeros((240, 320, 3), np.uint8)
        candidate_cropped = np.zeros((240, 320, 3), np.uint8)
    return candidate_img, candidate_cropped


def get_contours(img, img_contour, area_min, raw_img, data_green):
    """Function that finds the objects' contours, finds the candidate object and invokes functions
    that crop it into stand-alone image.

    Parameters:
    img (np.ndarray): Image preprocessed by `prepare_img` function.
    img_contour (np.ndarray): #TODO
    area_min (int): Minimal area of an object to become recognized.
    raw_img (np.ndarray): the original image from the drone.
    data_green (#TODO): trackbars' parameters for the green color recognition.
    """
    area = 0  # if contours not found, area = 0
    frame_h, frame_w = list(img.shape)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > area_min:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            xywh = cv2.boundingRect(approx)

            candidate_img, candidate_cropped = create_candidates(xywh, raw_img, cnt)
            candidate_for_contours = candidate_img.copy()

            area_min_green = data_green[-1]
            cand_dil, cand_res = prepare_img(data_green, candidate_img)
            correct_candidate = get_candidate_contours(cand_dil, candidate_for_contours, area_min_green)

            if not correct_candidate:
                no_candidate = np.zeros((240, 320, 3), np.uint8)
                candidate_cropped = np.zeros((240, 320, 3), np.uint8)
                stack_cand = stack_images(1, ([no_candidate, cand_res, img_contour], [cand_dil, candidate_cropped, np.zeros((240, 320, 3), np.uint8)]))
                cv2.imshow('candidate', stack_cand)

                area = 0
                return area

            elif correct_candidate:
                x, y, w, h = xywh
                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)

                stack_cand = stack_images(1, ([candidate_img, cand_res, img_contour], [cand_dil, candidate_cropped, np.zeros((240, 320, 3), np.uint8)]))
                cv2.imshow('candidate', stack_cand)

                cx = int(x + (w / 2))
                cy = int(y + (h / 2))

                cv2.circle(img, (int(frame_w / 2), int(frame_h / 2)), 5, (0, 0, 255), 5)
                cv2.line(img_contour, (int(frame_w / 2), int(frame_h / 2)), (cx, cy),
                         (0, 0, 255), 3)

                return area  # if correct candidate found, return the area
    return area

