from easytello_fix import PokeTello
import numpy as np
import cv2

width = 320
height = 240
fly = 0


drone = PokeTello()
print(drone.get_battery())

if fly:
    drone.takeoff()

    drone.land()

drone.streamoff()
drone.streamon()

while True:
    if drone.frame is not None:
        # cv2.imshow('Cap', drone.frame)
        img = cv2.resize(drone.frame, (width, height))
        cv2.imshow('Cap', img)

        # Video S`tream is closed if escape key is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

