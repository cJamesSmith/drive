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
import pygame
from scipy import spatial
import gym
from gym import spaces
from setup import setup
from simple_pid import PID
from do_mpc.controller import MPC
from do_mpc.model import Model


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
        self.lincoln = blueprint_library.filter("vehicle.*")[12]
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
        self.pid = PID(0.25, 0.01, 0.05, setpoint=20, output_limits=(-1, 1))
        self.neighbors = 3
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
            # 5348,
            28722,
            24052,
            19672,
            # 15500,
            # 8350,
            # 15500,
            # 8350,
            # 15500,
            # 8350,
            # 15500,
            # 8350,
            # 495,
            # 10958,
            # 29053,
            # 30286,
        ]
        # self.curve_point_id = [i * 1000 for i in range(0, 31)]
        self.sensor = blueprint_library.find("sensor.other.collision")
        self.tip_spawn_point = carla.Transform(carla.Location(x=2))
        self.cam_bp = blueprint_library.find("sensor.camera.rgb")
        self.IM_SHAPE = (1280, 720)  # width, height
        self.cam_bp.set_attribute("image_size_x", f"{self.IM_SHAPE[0]}")
        self.cam_bp.set_attribute("image_size_y", f"{self.IM_SHAPE[1]}")
        # self.cam_bp.set_attribute("fov", "80")
        self.cam_img = np.zeros((206, 97, 1))
        self.theta1_set = 0
        self.des_v = 30
        self.mpc = self.make_mpc()
        self.state_and_action = []
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.human_steer = 0

    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float)
        # return gym.spaces.Box(low=0, high=255, shape=(206, 97, 1), dtype=np.uint8)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        if self.action_type == "continuous":
            return gym.spaces.Box(low=0, high=0.1, shape=(1,))
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
        self.cam = self.world.spawn_actor(
            self.cam_bp, carla.Transform(carla.Location(x=-8, z=6.0), carla.Rotation(8.0, 0, 0)), attach_to=self.vehicle
        )
        self.sensor_list.append(self.cam)
        self.cam.listen(lambda data: self.process_img(data))
        self.actor_list.append(self.vehicle)
        self.sensor_list.append(self.tip)

        obs = self.get_obs()
        # print(f"************obs: {obs}****************")
        return obs
        # return self.cam_img

    def process_img(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((self.IM_SHAPE[1], self.IM_SHAPE[0], 4))
        # i = i[50:, 80:177, :3]
        self.cam_img = (0.2989 * i[:, :, 0] + 0.5870 * i[:, :, 1] + 0.1140 * i[:, :, 2]).astype(np.uint8)
        # self.cam_img = self.cam_img.reshape((206, 97, 1))
        # i /= 255.0
        # i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
        # self.cam_img = 0.2989 * i[:, :, 0] + 0.5870 * i[:, :, 1] + 0.1140 * i[:, :, 2]
        # print("image shape: ", i.shape)

    def step(self, action):
        self.world.tick()  # TODO: it must have, don't fuck
        # print(f"****************action: {action}******************")
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.human_steer = self.joystick.get_axis(0)
                print(self.human_steer)
        states = self.get_obs()
        # self.theta1["_p"] = action[0]
        # steer = self.mpc_control(states) + action[0] * self.human_steer
        steer = self.mpc_control(states) + action[0] * self.human_steer
        print(f"steer: {steer}")
        # obs = self.get_obs()
        # obs = None
        if self.cam_img is not None:
            cv2.imshow("", self.cam_img)
            # cv2.imwrite(f"{self.frame_step}.jpg", self.cam_img)
            key = cv2.waitKey(1)
            if key == 27:
                assert False
        self.frame_step += 1
        throttle = self.pid(self.v)
        # print(throttle)
        if throttle < 0:
            action = carla.VehicleControl(throttle=0.0, brake=-throttle, steer=float(steer))
        else:
            action = carla.VehicleControl(throttle=throttle, steer=float(steer), brake=0.0)

        logging.debug("{}, {}, {}".format(action.throttle, action.steer, action.brake))
        self.vehicle.apply_control(action)
        # if self.last_action == None:
        #     self.last_action = action

        done = False
        reward = 0
        info = dict()

        reward -= math.fabs(states[0]) * 10
        reward -= math.fabs(states[3])

        self.epi_err_dist += math.fabs(states[0])
        self.epi_err_yaw += math.fabs(states[3])

        if self.frame_step >= self.steps_per_episode:
            print(f"good one, frame: {states[:4], self.epi_err_dist / self.frame_step, self.epi_err_yaw / self.frame_step}")
            done = True
            # assert False

        return states, reward, done, info
        # return self.cam_img, reward, done, info

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

        yaw_err = math.radians(find_y[np.argmin(abs(find_y))])
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

        nearest_err = nearest_err * math.sin(angle_car_road)
        obs.append(nearest_err)
        obs.append((nearest_err - self.last_nearest_err) * self.fps)
        obs.append(yaw_err)
        obs.append((yaw_err - self.last_yaw_err) * self.fps)
        # obs.append(0 if self.last_action is None else self.last_action.steer * 10)
        v = self.vehicle.get_velocity()
        self.v = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        # obs.append(v - self.des_v)
        obs.append(self.human_steer)
        obs.append(curve)
        self.last_nearest_err = nearest_err
        self.last_yaw_err = yaw_err
        return obs

    def make_mpc(self):
        desired_speed = self.des_v
        T_all = 1 / self.fps
        cf = 4.92 * 10000
        cr = -3.115810432198605 * 10000

        mass = 2404
        lf = 1.471
        lr = 1.389
        IZ = 1536.7

        aa = -2 / mass / desired_speed
        bb = aa

        ccc = -2 * lf / mass / desired_speed
        dd = 2 * lr / mass / desired_speed
        ee = -2 * lf / IZ / desired_speed
        ff = 2 * lr / IZ / desired_speed
        gg = -2 * lf * lf / IZ / desired_speed
        hh = -2 * lr * lr / IZ / desired_speed
        ii = 2 / mass
        jjj = 2 * lf / IZ

        A_model = np.array(
            [
                [1, T_all, 0, 0],
                [
                    0,
                    T_all * (aa * cf + bb * cr) + 1,
                    -T_all * (aa * cf + bb * cr) * desired_speed,
                    T_all * ((ccc * cf + dd * cr)),
                ],
                [0, 0, 1, T_all],
                [
                    0,
                    T_all * (ee * cf + ff * cr),
                    -T_all * (ee * cf + ff * cr) * desired_speed,
                    1 + T_all * (gg * cf + hh * cr),
                ],
            ]
        )

        B_model = np.array([[0], [ii * T_all * cf], [0], [cf * jjj * T_all]])

        # np.savetxt("A.txt", A_model)
        # np.savetxt("B.txt", B_model)

        ##初始化MPC
        model_type = "discrete"
        model_mpc = Model(model_type)

        x0 = model_mpc.set_variable(var_type="_x", var_name="x0", shape=(1, 1))
        x1 = model_mpc.set_variable(var_type="_x", var_name="x1", shape=(1, 1))
        x2 = model_mpc.set_variable(var_type="_x", var_name="x2", shape=(1, 1))
        x3 = model_mpc.set_variable(var_type="_x", var_name="x3", shape=(1, 1))

        theta1 = model_mpc.set_variable("parameter", "theta1")

        u = model_mpc.set_variable(var_type="_u", var_name="u", shape=(1, 1))

        ##_------------------------------------------------------------------------------------------------------------------
        x0_n = x0 * A_model[0, 0] + x1 * A_model[0, 1]
        model_mpc.set_rhs("x0", x0_n)
        x0_n = x1 * A_model[1, 1] + x2 * A_model[1, 2] + x3 * A_model[1, 3] + u * B_model[1]
        model_mpc.set_rhs("x1", x0_n)
        x0_n = x2 * A_model[2, 2] + x3 * A_model[2, 3]
        model_mpc.set_rhs("x2", x0_n)
        x0_n = x1 * A_model[3, 1] + x2 * A_model[3, 2] + x3 * A_model[3, 3] + u * B_model[3]
        model_mpc.set_rhs("x3", x0_n)

        model_mpc.setup()
        mpc = MPC(model_mpc)
        # 设置控制器
        setup_mpc = {
            "n_robust": 10,
            "n_horizon": 10,
            "t_step": T_all,
            "state_discretization": "discrete",
            "store_full_solution": False,
            "nlpsol_opts": {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0},
        }
        mpc.set_param(**setup_mpc)

        mterm = 0.1 * model_mpc.x["x0"] ** 2 + 0.1 * model_mpc.x["x2"] ** 2
        lterm = (
            model_mpc.x["x0"]
            ** 2
            # + 100 * model_mpc.x["x2"] ** 2
            # + model_mpc.x["x3"] ** 2
        )
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=5)
        # mpc.set_rterm(u=theta1)

        mpc.bounds["lower", "_u", "u"] = -0.15
        mpc.bounds["upper", "_u", "u"] = 0.15

        self.theta1 = mpc.get_p_template(1)
        mpc.set_p_fun(self.get_theta)

        mpc.setup()
        return mpc

    def get_theta(self, time):
        # global theta1_set
        # self.theta1["_p"] = theta1_set
        return self.theta1

    def mpc_control(self, states):

        # print(states)
        # if math.fabs(states[0]) > 0.02:
        #     self.theta1["_p"] = 1
        # else:
        #     self.theta1["_p"] = 10
        # self.theta1["_p"] = 5
        x00 = np.array(states[:4]).T
        u0 = self.mpc.make_step(x00)
        return u0[0, 0]
