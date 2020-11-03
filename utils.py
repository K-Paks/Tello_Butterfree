import cv2.cv2 as cv2
import numpy as np


# test
def empty(a):
    pass


# class used to create trackbar window and get their parameters
class TrackbarWindow:
    def __init__(self, num, color=None):
        if color == 'white':  # predefined values for mapping whites only
            init_h_min, init_h_max, init_s_min, init_s_max, init_v_min, init_v_max = 0, 179, 0, 59, 220, 255
            init_thr1, init_thr2, init_area = 166, 171, 2000
        elif color == 'green':
            init_h_min, init_h_max, init_s_min, init_s_max, init_v_min, init_v_max = 55, 90, 30, 170, 40, 245
            init_thr1, init_thr2, init_area = 0, 255, 1000
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


# code by Murtaza's Workshop (Youtube)
def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        # hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def prepare_img(data, img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, area_min = data

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    img_blur = cv2.GaussianBlur(result, (5, 5), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    img_dil = cv2.dilate(img_canny, kernel, iterations=2)
    return img_dil, result


def get_candidate_contours(img, img_contour, area_min):
    area = 0
    # crop_xywh = None

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_min:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            # crop_xywh = x, y, w, h

            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)

    if area is not 0:
        return 1
    else:
        return 0


def get_contours(raw_img, img, img_contour, frame_w, frame_h, dead_zone, area_min, green_trackbars):
    direction = 0
    area = 0
    crop_xywh = None

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dead_zone_w = dead_zone
    dea_zone_h = dead_zone - 15

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > area_min:

            cv2.drawContours(img_contour, contours, -1, (255, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            crop_xywh = x, y, w, h

            mask = np.ones(raw_img.shape[:2])
            mask = cv2.drawContours(mask, [cnt], -1, 0, cv2.FILLED)
            candidate_img2 = raw_img.copy()
            candidate_img2[mask.astype(np.bool), :] = 0
            candidate_img2 = cv2.resize(candidate_img2, (320, 240))

            if crop_xywh is not None:
                x, y, w, h = crop_xywh
                candidate_img = raw_img[y:y + h, x:x + w]
                candidate_img = cv2.resize(candidate_img, (320, 240))
            else:
                candidate_img = np.zeros((240, 320, 3), np.uint8)

            cand_contour = candidate_img.copy()
            data_green = green_trackbars.get_trackbar_values()
            area_min_green = data_green[-1]

            cand_dil, cand_res = prepare_img(data_green, candidate_img)
            correct_candidate = get_candidate_contours(cand_dil, cand_contour, area_min_green)

            if not correct_candidate:
                no_candidate = np.zeros((240, 320, 3), np.uint8)
                stack_cand = stack_images(1, ([no_candidate, cand_res], [cand_dil, candidate_img2]))
                cv2.imshow('candidate', stack_cand)

            elif correct_candidate:
                stack_cand = stack_images(1, ([candidate_img, cand_res], [cand_dil, candidate_img2]))
                cv2.imshow('candidate', stack_cand)

                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)

                # cv2.putText(img_contour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                #             (0, 255, 0), 2)
                # cv2.putText(img_contour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                #             (0, 255, 0), 2)
                # print(area)
                # cv2.putText(img_contour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,
                #             0.7,
                #             (0, 255, 0), 2)

                cx = int(x + (w / 2))
                cy = int(y + (h / 2))
                # print(cx, cy)

                # if cx < int(frame_w / 2) - dead_zone_w:
                #     cv2.putText(img_contour, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                #     cv2.rectangle(img_contour, (0, int(frame_h / 2 - dea_zone_h)),
                #                   (int(frame_w / 2) - dead_zone_w, int(frame_h / 2) + dea_zone_h), (0, 0, 255),
                #                   cv2.FILLED)
                #     direction = 1
                # elif cx > int(frame_w / 2) + dead_zone_w:
                #     cv2.putText(img_contour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                #     cv2.rectangle(img_contour, (int(frame_w / 2 + dead_zone_w), int(frame_h / 2 - dea_zone_h)),
                #                   (frame_w, int(frame_h / 2) + dea_zone_h), (0, 0, 255), cv2.FILLED)
                #     direction = 2
                # elif cy < int(frame_h / 2) - dea_zone_h:
                #     cv2.putText(img_contour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                #     cv2.rectangle(img_contour, (int(frame_w / 2 - dead_zone_w), 0),
                #                   (int(frame_w / 2 + dead_zone_w), int(frame_h / 2) - dea_zone_h), (0, 0, 255),
                #                   cv2.FILLED)
                #     direction = 3
                # elif cy > int(frame_h / 2) + dea_zone_h:
                #     cv2.putText(img_contour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                #     cv2.rectangle(img_contour, (int(frame_w / 2 - dead_zone_w), int(frame_h / 2) + dea_zone_h),
                #                   (int(frame_w / 2 + dead_zone_w), frame_h), (0, 0, 255), cv2.FILLED)
                #     direction = 4
                # else:
                #     direction = 0

                cv2.line(img_contour, (int(frame_w / 2), int(frame_h / 2)), (cx, cy),
                         (0, 0, 255), 3)

                return direction, area, crop_xywh
    return direction, area, crop_xywh


def display(img, frame_w, frame_h, dead_zone):
    dead_zone_w = dead_zone
    dea_zone_h = dead_zone - 15
    cv2.line(img, (int(frame_w / 2) - dead_zone_w, 0), (int(frame_w / 2) - dead_zone_w, frame_h), (255, 255, 0),
             3)
    cv2.line(img, (int(frame_w / 2) + dead_zone_w, 0), (int(frame_w / 2) + dead_zone_w, frame_h), (255, 255, 0),
             3)

    cv2.circle(img, (int(frame_w / 2), int(frame_h / 2)), 5, (0, 0, 255), 5)
    cv2.line(img, (0, int(frame_h / 2) - dea_zone_h), (frame_w, int(frame_h / 2) - dea_zone_h), (255, 255, 0),
             3)
    cv2.line(img, (0, int(frame_h / 2) + dea_zone_h), (frame_w, int(frame_h / 2) + dea_zone_h), (255, 255, 0),
             3)
