import atexit
import logging
import math
import os
import signal
import carla
import random
import time
import cv2
import numpy as np
from scipy import spatial
import gym
from gym import spaces
from setup import setup
from simple_pid import PID


class CarlaEnv(gym.Env):
    def __init__(
        self,
        town,
        fps,
        im_width,
        im_height,
        repeat_action,
        start_transform_type,
        sensors,
        action_type,
        enable_preview,
        steps_per_episode,
        playing=False,
        timeout=10,
    ):
        # Env initialization
        self.client, self.world, self.frame, self.server = setup(town=town, fps=fps, client_timeout=timeout, playing=playing)
        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        self.lincoln = blueprint_library.filter("lincoln")[0]
        self.repeat_action = repeat_action
        self.action_type = action_type
        self.start_transform_type = start_transform_type
        self.actor_list = []
        self.sensor_list = []
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.my_waypoint = np.loadtxt("my_waypoint")
        self.tree = spatial.KDTree(self.my_waypoint[:, :3])
        self.nearest = None
        self.start_transform_Town04 = carla.Transform(
            carla.Location(x=self.my_waypoint[1000][0], y=self.my_waypoint[1000][1], z=self.my_waypoint[1000][2] + 0.1),
            carla.Rotation(0, self.my_waypoint[1000][-1], 0),
        )

        print(self.start_transform_Town04)
        self.last_nearest_err = 0
        self.last_yaw_err = 0
        self.pid = PID(0.25, 0.01, 0.05, setpoint=30, output_limits=(-1, 1))
        self.neighbors = 5
        self.last_nearest_id = 0
        self.nearest_id = 0
        self.fps = fps
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(self.settings)
        self.last_action = None
        self.snapshot = self.world.get_snapshot()
        self.epi_err_dist = 0
        self.epi_err_yaw = 0
        self.curve_point_id = [
            5348,
            28722,
            24052,
            19672,
            15500,
            8350,
            15500,
            8350,
            15500,
            8350,
            15500,
            8350,
            # 495,
            #    10958, 29053, 30286
        ]
        # self.curve_point_id = [i * 1000 for i in range(0, 31)]
        self.sensor = blueprint_library.find("sensor.other.collision")
        self.tip_spawn_point = carla.Transform(carla.Location(x=2))
        self.cam_bp = blueprint_library.find("sensor.camera.rgb")
        self.cam_bp.set_attribute("image_size_x", f"{1280}")
        self.cam_bp.set_attribute("image_size_y", f"{720}")
        self.cam_bp.set_attribute("fov", "110")
        self.cam_img = None

    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=-3, high=3, shape=(6,), dtype=np.float)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        if self.action_type == "continuous":
            return gym.spaces.Box(low=-0.1, high=0.1, shape=(1,))
        elif self.action_type == "discrete":
            return gym.spaces.MultiDiscrete([4, 9])
        else:
            raise NotImplementedError()
        # TODO: Add discrete actions (here and anywhere else required)

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return seed

    def reset(self):
        self.world.tick()  # TODO: it must have, don't fuck
        self._destroy_agents()
        # Car, sensors, etc. We create them every episode then destroy
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.frame_step = 0
        self.out_of_loop = 0
        self.dist_from_start = 0
        self.epi_err_dist = 0
        self.epi_err_yaw = 0
        self.last_action = None
        id_rand = random.choice(self.curve_point_id)
        xy_yaw_rand = self.my_waypoint[id_rand]
        self.last_loc = carla.Transform(
            carla.Location(x=xy_yaw_rand[0], y=xy_yaw_rand[1], z=xy_yaw_rand[2] + 0.1), carla.Rotation(0, xy_yaw_rand[3], 0)
        )
        # self.last_loc = self.start_transform_Town04
        # print(self.last_loc)
        self.vehicle = self.world.try_spawn_actor(self.lincoln, self.last_loc)
        while self.vehicle is None:
            id_rand = random.choice(self.curve_point_id)
            xy_yaw_rand = self.my_waypoint[id_rand]
            self.last_loc = carla.Transform(
                carla.Location(x=xy_yaw_rand[0], y=xy_yaw_rand[1], z=xy_yaw_rand[2] + 0.1), carla.Rotation(0, xy_yaw_rand[3], 0)
            )
            self.vehicle = self.world.try_spawn_actor(self.lincoln, self.last_loc)
        self.start_transform_Town04 = self.last_loc
        self.tip = self.world.spawn_actor(self.sensor, self.tip_spawn_point, attach_to=self.vehicle)
        if self.playing:
            self.cam = self.world.spawn_actor(self.cam_bp, carla.Transform(carla.Location(x=-4, z=2)), attach_to=self.vehicle)
            self.sensor_list.append(self.cam)
            self.cam.listen(lambda data: self.process_img(data))
        self.actor_list.append(self.vehicle)
        self.sensor_list.append(self.tip)

        # Just to make it start recording, apparently passing an empty command makes it react
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(0.1)
        # self.vehicle.set_target_velocity(carla.Vector3D(10, 0, 0))
        # print(self.vehicle.get_transform())
        # observation:
        return self.get_obs()

    def step(self, action):
        self.world.tick()  # TODO: it must have, don't fuck
        total_reward = 0
        for _ in range(self.repeat_action):
            obs, rew, done, info = self._step(action)
            total_reward += rew
            if done:
                break
        return obs, total_reward, done, info

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(dir(image))
        # print(i.shape)
        i2 = i.reshape((720, 1280, 4))
        i3 = i2[:, :, :3]
        self.cam_img = i3

    def _step(self, action):
        # if self.playing:
        #     self.cam.listen(lambda data: self.process_img(data))
        if self.cam_img is not None:
            cv2.imshow("", self.cam_img)
            key = cv2.waitKey(20)
            if key == 27:
                assert False
        # if math.fabs(action[0]) < 0.02:
        #     action[0] = 0
        self.frame_step += 1
        # print(self.world.get_snapshot().frame)

        # Apply control to the vehicle based on an action

        # Calculate speed in km/h from car's velocity (3D vector)
        # action = carla.Ve#hicleControl(throttle=0.5, steer=float(action[0]), brake=0)
        v = self.vehicle.get_velocity()
        v = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        if self.action_type == "continuous":
            throttle = self.pid(v)
            # throttle = 0.5
            # print(throttle)
            # print(action)
            if throttle < 0:
                action = carla.VehicleControl(throttle=0.0, brake=-throttle, steer=float(action[0]))
            else:
                action = carla.VehicleControl(throttle=throttle, steer=float(action[0]), brake=0.0)
            # action = carla.VehicleControl(throttle=0.5, steer=float(action[0]), brake=0)
        elif self.action_type == "discrete":
            if action[0] == 0:
                action = carla.VehicleControl(throttle=0, steer=float((action[1] - 4) / 4), brake=1)
            else:
                action = carla.VehicleControl(throttle=float((action[0]) / 3), steer=float((action[1] - 4) / 4), brake=0)
        else:
            raise NotImplementedError()
        logging.debug("{}, {}, {}".format(action.throttle, action.steer, action.brake))
        self.vehicle.apply_control(action)
        if self.last_action == None:
            self.last_action = action
        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform_Town04.location)
        square_dist_diff = new_dist_from_start**2 - self.dist_from_start**2
        self.dist_from_start = new_dist_from_start

        done = False
        reward = 0
        info = dict()

        obs = self.get_obs()
        reward -= math.fabs(obs[0])
        # reward -= obs[1]
        reward -= math.fabs(obs[2])
        # reward -= obs[3]
        # reward += square_dist_diff
        reward += 8
        # reward -= math.fabs(action.steer - self.last_action.steer)
        self.last_action = action
        # reward -= math.fabs(action.steer)
        # reward += 10 * (
        #     (self.my_waypoint[self.last_nearest_id][0] - self.my_waypoint[self.nearest_id][0]) ** 2
        #     + (self.my_waypoint[self.last_nearest_id][1] - self.my_waypoint[self.nearest_id][1]) ** 2
        # )

        self.epi_err_dist += math.fabs(obs[0])
        self.epi_err_yaw += math.fabs(obs[2])

        # 训练时改小，测试时改大一点
        if math.fabs(obs[0]) > 2 or 20 > math.fabs(obs[2]) > 10 + math.fabs(obs[-1]):
            print(
                f"fucked {obs}, {self.my_waypoint[self.nearest_id], self.epi_err_dist / self.frame_step, self.epi_err_yaw / self.frame_step}, l={self.frame_step}"
            )
            done = True
            reward -= 500

        if self.frame_step >= self.steps_per_episode:
            print(f"good one, frame: {self.my_waypoint[self.nearest_id], self.epi_err_dist / self.frame_step, self.epi_err_yaw / self.frame_step}")
            done = True
            reward += 500

        # self.total_reward += reward

        if done:
            logging.debug("Env lasts {} steps, restarting ... ".format(self.frame_step))
            self._destroy_agents()

        return obs, reward, done, info

    def close(self):
        logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))

    def _destroy_agents(self):
        for sensor in self.sensor_list:
            if sensor is not None and sensor.is_alive:
                # sensor.stop()
                sensor.destroy()
        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, "is_listening") and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def get_obs(self):
        obs = []
        cur_pos = [self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y, self.vehicle.get_transform().location.z]
        cur_yaw = self.vehicle.get_transform().rotation.yaw
        tip_pos = [self.tip.get_transform().location.x, self.tip.get_transform().location.y, self.tip.get_transform().location.z]
        tip_nearest = self.tree.query(tip_pos, p=2)
        tip_nearest_yaw = self.my_waypoint[tip_nearest[1]][-1]

        nearest = self.tree.query(cur_pos, k=self.neighbors, p=2)
        self.nearest_id = nearest[1][0]
        way_yaw = self.my_waypoint[self.nearest_id][-1]
        find_y = np.array(
            [
                cur_yaw - way_yaw,
                cur_yaw - way_yaw + 360,
                cur_yaw - way_yaw - 360,
            ]
        )

        find_curve = np.array(
            [
                tip_nearest_yaw - way_yaw,
                tip_nearest_yaw - way_yaw + 360,
                tip_nearest_yaw - way_yaw - 360,
            ]
        )
        curve = find_curve[np.argmin(abs(find_curve))]

        way_yaw = math.radians(way_yaw)

        yaw_err = find_y[np.argmin(abs(find_y))]
        nearest_err = nearest[0][0]

        angle_car_road = math.atan2(self.my_waypoint[self.nearest_id][1] - cur_pos[1], self.my_waypoint[self.nearest_id][0] - cur_pos[0]) - way_yaw
        find_y = np.array(
            [
                angle_car_road,
                angle_car_road + 2 * math.pi,
                angle_car_road - 2 * math.pi,
            ]
        )
        angle_car_road = find_y[np.argmin(abs(find_y))]

        obs.append(nearest_err * math.sin(angle_car_road))
        obs.append((nearest_err - self.last_nearest_err) * 10)
        obs.append(yaw_err)
        obs.append(yaw_err - self.last_yaw_err)
        obs.append(0 if self.last_action is None else self.last_action.steer * 10)
        obs.append(curve)
        self.last_nearest_err = nearest_err
        self.last_yaw_err = yaw_err
        return obs
