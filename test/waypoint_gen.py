import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import pygame
import matplotlib.pyplot as plt
import math

IM_WIDTH = 640
IM_HEIGHT = 480

import carla

actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    map = world.get_map()
    waypoint_list = map.generate_waypoints(0.1)
    lane_id_set = set()
    for way in waypoint_list:
        lane_id_set.add(way.lane_id)
    print(lane_id_set)

    LANE_ID = 3

    my_waypoint = []

    for way in waypoint_list:
        if way.lane_id == LANE_ID:
            # print(way.transform)
            # break
            x = way.transform.location.x
            y = way.transform.location.y
            yaw = way.transform.rotation.yaw
            yaw = math.fmod(yaw, 360)
            if yaw < 0:
                yaw += 360

            my_waypoint.append([x, y, yaw])
    my_waypoint = np.array(my_waypoint)
    print(my_waypoint.shape)
    np.savetxt("my_waypoint", my_waypoint)  # save
    # plt.plot(my_waypoint[:, 0], my_waypoint[:, 1], "o")  # visualize
    # plt.show()


finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up")
    pass
