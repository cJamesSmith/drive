"""
Welcome to CARLA manual control.
"""

from __future__ import print_function

import os
import sys

import os

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from simple_pid import PID
from scipy import spatial


from do_mpc.controller import MPC
from do_mpc.model import Model

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_k
    from pygame.locals import K_j
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        self.map = self.world.get_map()

        self.hud = hud
        self.player = None
        # self.imu_sensor = None
        self.camera_manager = None
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All,
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        vehicle_c = self.world.get_blueprint_library().filter(self._actor_filter)
        blueprint = vehicle_c[12]
        blueprint.set_attribute("role_name", self.actor_role_name)
        if blueprint.has_attribute("color"):
            color = blueprint.get_attribute("color").recommended_values
            color = color[1]
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")
        # set the max speed
        if blueprint.has_attribute("speed"):
            self.player_max_speed = float(blueprint.get_attribute("speed").recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute("speed").recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 0.1
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print("There are no spawn points available in your map/town.")
                print("Please add some Vehicle Spawn Point to your UE4 scene.")
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            ##--------------------------------------------------------------------------------
            spawn_point = spawn_points[271]
            spawn_point.location.x += 3.5
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification("LayerMap selected: %s" % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification("Unloading map layer: %s" % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification("Loading map layer: %s" % selected)
            self.world.load_map_layer(selected)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._carsim_enabled = False
        self._carsim_road = False
        self._autopilot_enabled = start_in_autopilot

        ##---------------i------------------
        self.remember = 0
        self.flag = True
        self.xunhuan = 0
        # self.x0_old = 0
        self.erryd = 0

        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        ##--------------------------------------------------------------------------------------------------

        self.pid = PID(0.25, 0.01, 0.05, setpoint=desired_speed, output_limits=(-1, 1))
        self.waypoints = np.loadtxt("my_waypoint")
        self.tree = spatial.KDTree(self.waypoints[:, :3])

        # pygame.joystick.init()
        # joystick_count = pygame.joystick.get_count()
        # if joystick_count > 1:
        #     raise ValueError("Please Connect Just One Joystick")
        # self._joystick = pygame.joystick.Joystick(0)
        # self._joystick.init()

    ##--------------------------------------------------------------------------------------------------

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()

        if not self._autopilot_enabled:
            ##---------------PID_Controller_For_Throttle--------------------##
            t = world.player.get_transform()
            v = world.player.get_velocity()
            v = (v.x**2 + v.y**2 + v.z**2) ** 0.5
            self._control.throttle = self.pid(v)
            ##---------------Mpc_Controller_For_Steer--------------------##

            cur_pos = [t.location.x, t.location.y, t.location.z]
            cur_yaw = t.rotation.yaw
            nearest = self.tree.query(cur_pos)

            way_yaw = self.waypoints[nearest[1], -1]
            way_x = self.waypoints[nearest[1], 0]
            way_y = self.waypoints[nearest[1], 1]

            global x_way_x
            global x_way_y
            global x_way_yaw

            x_way_x = way_x
            x_way_y = way_y
            x_way_yaw = way_yaw

            x_use = t.location.x
            y_use = t.location.y
            y_yaw = t.rotation.yaw

            TTT = T_all

            way_yaw = math.radians(way_yaw)
            y_yaw = math.radians(y_yaw)

            find_y = np.array(
                [
                    y_yaw - way_yaw,
                    y_yaw - way_yaw + 2 * math.pi,
                    y_yaw - way_yaw - 2 * math.pi,
                ]
            )
            index_find = np.argmin(abs(find_y))
            err_yaw = find_y[index_find]

            if -math.sin(way_yaw) * (x_use - way_x) + math.cos(way_yaw) * (y_use - way_y) > 0:
                sig = -1
            else:
                sig = 1

            # err_d = nearest[0] * sig
            angle_car_road = math.atan2(way_y - y_use, way_x - x_use) - way_yaw
            find_y = np.array(
                [
                    angle_car_road,
                    angle_car_road + 2 * math.pi,
                    angle_car_road - 2 * math.pi,
                ]
            )
            angle_car_road = find_y[np.argmin(abs(find_y))]
            err_d = nearest[0] * math.sin(angle_car_road)

            if self.xunhuan > 2:
                err_d_d = (err_d - self.rem_d) / (TTT)
                err_y_d = (err_yaw - self.remrem_y) / (TTT)

                if abs(err_y_d) > 10:
                    err_y_d = self.erryd

                # ##sb滤波
                # err_d = 0.5 * err_d + 0.5 * self.rem_d
                self.rem_d = err_d
                self.remrem_d = self.rem_d
                self.rem_y = err_yaw
                self.remrem_y = self.rem_y

                # err_y_d += self.erryd*0.5

                self.erryd = err_y_d

            else:
                err_d_d = 0
                err_y_d = 0
                self.rem_d = err_d
                self.remrem_d = err_d
                self.rem_y = err_yaw
                self.remrem_y = err_yaw
            ##注意这里为了简化多加了个way_yaw
            global x00
            x00 = np.array([err_d, err_d_d, err_yaw, err_y_d]).T

            if self.flag == True:
                self.flag = False
                mpc.x0 = x00
                mpc.set_initial_guess()

            ##mpc控制
            x0_save.append(x00)
            ##滤波
            u0 = mpc.make_step(x00)
            # print(u0)
            self._control.steer = u0[0, 0]

            # if math.fabs(self._control.steer) < 0.01:
            #     self._control.steer = 0

            if self._control.throttle > 0:
                self._control.brake = 0

            else:
                self._control.throttle = 0
                self._control.brake = -self._control.throttle
            self.xunhuan += 1
            world.player.apply_control(self._control)

    @staticmethod
    def _is_quit_shortcut(key):
        if (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL):
            return True
        else:
            return False


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = "courier" if os.name == "nt" else "mono"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        vehicles = world.world.get_actors().filter("vehicle.*")
        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(world.player, truncate=20),
            "Map:     % 20s" % world.map.name.split("/")[-1],
            "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (t.location.x, t.location.y)),
            "Height:  % 18.0f m" % t.location.z,
            "",
        ]

        # database_save.append(
        #     [v.x, v.y, world.imu_sensor.gyroscope[2], t.location.x, t.location.y, c.steer, self.simulation_time,
        #      world.imu_sensor.accelerometer[0], t.rotation.yaw, world.imu_sensor.accelerometer[1],
        #      x_way_x, x_way_y,
        #      x_way_yaw])  # 再说吧 找找speed_X实在不行就积分，应该也对

        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", c.throttle, 0.0, 1.0),
                ("Steer:", c.steer, -1.0, 1.0),
                ("Brake:", c.brake, 0.0, 1.0),
                ("Reverse:", c.reverse),
                ("Hand brake:", c.hand_brake),
                ("Manual:", c.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(c.gear, c.gear),
            ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [("Speed:", c.speed, 0.0, 5.556), ("Jump:", c.jump)]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            distance = lambda l: math.sqrt((l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (
                carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)),
                Attachment.SpringArm,
            ),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (
                carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
                Attachment.SpringArm,
            ),
            (
                carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)),
                Attachment.SpringArm,
            ),
            (
                carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)),
                Attachment.Rigid,
            ),
        ]
        self.transform_index = 1
        self.sensors = [["sensor.camera.rgb", cc.Raw, "Camera RGB", {}]]  # ,
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(hud.dim[0]))
                bp.set_attribute("image_size_y", str(hud.dim[1]))
                if bp.has_attribute("gamma"):
                    bp.set_attribute("gamma", str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith("sensor.lidar"):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == "range":
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notificaKeyboardControltion("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith("sensor.camera.dvs"):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data,
                dtype=np.dtype(
                    [
                        ("x", np.uint16),
                        ("y", np.uint16),
                        ("t", np.int64),
                        ("pol", np.bool),
                    ]
                ),
            )
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        settings = world.world.get_settings()
        settings.fixed_delta_seconds = T_all
        settings.synchronous_mode = True
        world.world.apply_settings(settings)
        ##-----------------------------------------------------------------

        clock = pygame.time.Clock()

        while True:
            # clock.tick_busy_loop(60)
            world.world.tick()
            if controller.parse_events(client, world, clock):
                settings = world.world.get_settings()
                settings.fixed_delta_seconds = None
                settings.synchronous_mode = False
                world.world.apply_settings(settings)
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument("-a", "--autopilot", action="store_true", help="enable autopilot")
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.*",
        help='actor filter (default: "vehicle.*")',
    )
    argparser.add_argument(
        "--rolename",
        metavar="NAME",
        default="hero",
        help='actor role name (default: "hero")',
    )
    argparser.add_argument(
        "--gamma",
        default=2.2,
        type=float,
        help="Gamma correction of the camera (default: 2.2)",
    )
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)
    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    desired_speed = 20
    T_all = 0.05
    # way_point_save = np.zeros([1,3])
    x_way_x = 0
    x_way_y = 0
    x_way_yaw = 0
    database_save = []
    x0_save = []
    x00 = np.zeros([1, 4])
    u_steer = 0

    cf = 4.92 * 10000
    cr = -3.115810432198605 * 10000

    mass = 2404
    lf = 1.471
    lr = 1.389
    IZ = 1536.7
    # vx = 10

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

    np.savetxt("A.txt", A_model)
    np.savetxt("B.txt", B_model)

    ##初始化MPC
    # t1 = time.time()
    model_type = "discrete"
    model_mpc = Model(model_type)

    x0 = model_mpc.set_variable(var_type="_x", var_name="x0", shape=(1, 1))
    x1 = model_mpc.set_variable(var_type="_x", var_name="x1", shape=(1, 1))
    x2 = model_mpc.set_variable(var_type="_x", var_name="x2", shape=(1, 1))
    x3 = model_mpc.set_variable(var_type="_x", var_name="x3", shape=(1, 1))

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
        "n_robust": 1,
        "n_horizon": 10,
        "t_step": T_all,
        "state_discretization": "discrete",
        "store_full_solution": False,
        "nlpsol_opts": {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0},
    }
    mpc.set_param(**setup_mpc)

    # mterm = model_mpc.x["x1"] ** 2
    mterm = 0.01 * model_mpc.x["x0"] ** 2 + 0.0004 * model_mpc.x["x1"] ** 2
    lterm = (
        model_mpc.x["x0"] ** 2
        + model_mpc.u["u"] ** 2
        # + 100 * model_mpc.x["x2"] ** 2
        # + model_mpc.x["x3"] ** 2
    )
    mpc.set_objective(mterm=mterm, lterm=lterm)
    # mpc.set_rterm(
    #     phi_m_1_set=1e-2,
    #     phi_m_2_set=1e-2
    # )

    mpc.bounds["lower", "_u", "u"] = -0.06
    mpc.bounds["upper", "_u", "u"] = 0.06

    mpc.setup()
    print("done!")

    main()
