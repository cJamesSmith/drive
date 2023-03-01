import atexit
import logging
import math
import os
import signal
import carla
import random
import time
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
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.my_waypoint = np.loadtxt("my_waypoint")
        self.tree = spatial.KDTree(self.my_waypoint[:, :2])
        self.nearest = None
        self.start_transform_Town04 = carla.Transform(
            carla.Location(x=self.my_waypoint[1000][0], y=self.my_waypoint[1000][1], z=0.1), carla.Rotation(0, self.my_waypoint[1000][2], 0)
        )

        print(self.start_transform_Town04)
        self.last_nearest_err = 0
        self.last_yaw_err = 0
        self.pid = PID(0.25, 0.01, 0.05, setpoint=10, output_limits=(-1, 1))
        self.neighbors = 20
        self.last_nearest_id = 0
        self.nearest_id = 0
        self.fps = fps
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(self.settings)
        self.last_action = None
        self.snapshot = self.world.get_snapshot()

    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float)

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
        self.last_loc = self.start_transform_Town04.location

        self.vehicle = self.world.spawn_actor(self.lincoln, self.start_transform_Town04)
        self.actor_list.append(self.vehicle)

        # Just to make it start recording, apparently passing an empty command makes it react
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(0.1)
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

    def _step(self, action):
        self.frame_step += 1
        # print(self.world.get_snapshot().frame)

        # Apply control to the vehicle based on an action

        # Calculate speed in km/h from car's velocity (3D vector)
        # action = carla.VehicleControl(throttle=0.5, steer=float(action[0]), brake=0)
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
        square_dist_diff = new_dist_from_start - self.dist_from_start
        self.dist_from_start = new_dist_from_start

        done = False
        reward = 0
        info = dict()

        obs = self.get_obs()
        reward -= obs[0]
        reward -= obs[1]
        reward -= math.fabs(obs[2])
        reward -= obs[3]
        self.last_action = action
        # reward += square_dist_diff
        reward += 10
        reward -= (action.steer - self.last_action.steer) ** 2
        reward -= math.fabs(action.steer)
        # reward += 10 * (
        #     (self.my_waypoint[self.last_nearest_id][0] - self.my_waypoint[self.nearest_id][0]) ** 2
        #     + (self.my_waypoint[self.last_nearest_id][1] - self.my_waypoint[self.nearest_id][1]) ** 2
        # )
        # self.last_loc = loc
        # reward += 1

        if math.fabs(obs[0]) > 0.1 or math.fabs(obs[2]) > 1:
            print(f"fucked obs: {obs}, {self.my_waypoint[self.nearest_id]}")
            done = True
            reward -= 100

        if self.frame_step >= self.steps_per_episode:
            print(f"*****good one, frame: {self.frame_step}*****")
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
        cur_pos = [self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y]
        cur_yaw = self.vehicle.get_transform().rotation.yaw
        cur_yaw = math.fmod(cur_yaw, 360)
        if cur_yaw < 0:
            cur_yaw += 360
        nearest = self.tree.query(cur_pos, k=self.neighbors, p=2)
        self.nearest_id = nearest[1][0]
        yaw_err = cur_yaw - self.my_waypoint[nearest[1][0]][-1]
        yaws = [self.my_waypoint[id][-1] for id in nearest[1]]
        curve = sum(yaws) / len(yaws)
        if yaw_err > 180:
            yaw_err = 360 - yaw_err
        curve_err = cur_yaw - curve
        if curve_err > 180:
            curve_err = 360 - curve_err
        nearest_err = nearest[0][0]
        obs.append(nearest_err)
        obs.append(nearest_err - self.last_nearest_err)
        obs.append(yaw_err)
        obs.append(yaw_err - self.last_yaw_err)
        # obs.append(curve_err / 360)
        # obs.append(self.vehicle.get_velocity())
        self.last_nearest_err = nearest_err
        self.last_yaw_err = yaw_err
        return obs
