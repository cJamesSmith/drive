import glob
import os
import sys
import random
import time
import numpy as np
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480


def process_img(image):
    i = np.array(image.raw_data)
    # print(dir(image))
    # print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    print(i3.shape)
    # imwrite performs well
    # cv2.imwrite(f"{image.frame}.jpg", i3)
    # TODO imshow failed
    # cv2.imshow("123", i3 / 255.0)
    # cv2.waitKey(1000 // 60)
    # cv2.destroyAllWindows()
    return i3 / 255.0


import carla

actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())
    print(spawn_point)
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    # vehicle.apply_control(carla.VehicleControl(throttle=1, steer=0.0))
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_img(data))

    time.sleep(5)

    pass
finally:
    cam_bp.destroy()
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up")
    pass
