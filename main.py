from easytello_fix import PokeTello





drone = PokeTello()

drone.takeoff()

print(drone.get_battery())

drone.get_acceleration()
drone.get_speed()


drone.rc_control(0, 10, 0, 0)
# time.sleep(3)
drone.rc_control(10, 0, 0, 0)



drone.land()

