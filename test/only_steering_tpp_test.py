import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import pygame
from scipy import spatial
import math
from simple_pid import PID

IM_WIDTH = 640
IM_HEIGHT = 480

import carla

actor_list = []

pid = PID(0.25, 0.01, 0.05, setpoint=20, output_limits=(-1, 1))

try:
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    print(joystick_count)
    if joystick_count > 1:
        raise ValueError("Please Connect Just One Joystick")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    my_waypoint = np.loadtxt("my_waypoint")
    tree = spatial.KDTree(my_waypoint[:, :2])
    bp = blueprint_library.filter("model3")[0]
    print(bp)

    yaws = my_waypoint[:, -1]
    # print(max(yaws), min(yaws))
    # sys.exit()

    # spawn_point = random.choice(world.get_map().get_spawn_points())\
    print(world.get_map().get_spawn_points())
    spawn_point = carla.Transform(carla.Location(x=my_waypoint[1000][0], y=my_waypoint[1000][1], z=0.1), carla.Rotation(0, my_waypoint[1000][2], 0))
    print(spawn_point)
    vehicle = world.spawn_actor(bp, spawn_point)
    time.sleep(0.1)
    print(vehicle.get_transform())
    # vehicle.set_autopilot(True)
    # vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.1))
    actor_list.append(vehicle)
    # print(vehicle.get_transform())
    # assert False
    while True:
        # vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.1))
        v = vehicle.get_velocity()
        v = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        throttle = pid(v)
        cur_pos = [vehicle.get_transform().location.x, vehicle.get_transform().location.y]
        cur_yaw = vehicle.get_transform().rotation.yaw
        cur_yaw = math.fmod(cur_yaw, 360)
        if cur_yaw < 0:
            cur_yaw += 360
        nearest = tree.query(cur_pos, p=2)
        yaw_dir = cur_yaw - my_waypoint[nearest[1]][-1]
        if 240 > math.fabs(yaw_dir) > 80:
            print("dir_err > 80 fucking shit")
            print(f"cur_pos: {cur_pos}, cur_yaw: {cur_yaw}, way_yaw: {my_waypoint[nearest[1]][-1]}")
            break
        # print(nearest[0], yaw_dir)

        time.sleep(1 / 30)
        for event in pygame.event.get():
            steer = 0
            if event.type == pygame.JOYAXISMOTION:
                steer = joystick.get_axis(0)
                print(steer)
            elif event.type == pygame.JOYBUTTONDOWN:
                if joystick.get_button(0):
                    print(joystick.get_button(0))
                elif joystick.get_button(1):
                    print(joystick.get_button(1))
            # vehicle.set_autopilot(True)
        if throttle < 0:
            vehicle.apply_control(carla.VehicleControl(throttle=0, brake=-throttle, steer=0))
        else:
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0, brake=0))
        print(throttle, v)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up")
    pass
