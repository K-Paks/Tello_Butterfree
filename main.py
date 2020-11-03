from easytello_fix import PokeTello
from utils import *
#test3
width = 320
height = 240

fly = 1

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


        data_white = white_trackbars.get_trackbar_values()  # returns: (h_min, h_max, s_min, s_max, v_min, v_max, threshold1, threshold2, areaMin)
        data_green = green_trackbars.get_trackbar_values()
        areaMin_white = data_white[-1]

        imgDil, result = prepare_img(data_white, img)

        area, hat_xy = get_candidate(img, imgDil, imgContour, areaMin_white, data_green)
        img_xy = (img.shape[1]/2, img.shape[0]/2)

        # candidate processing
        stack = stack_images(1, ([img, result], [imgDil, imgContour]))
        cv2.imshow('Horizontal Stacking', stack)



        # steering
        if sum(hat_xy) >= 0:
            follow_candidate(drone, img_xy, hat_xy, area)




        # Video Stream is closed if escape key is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
drone.land()
