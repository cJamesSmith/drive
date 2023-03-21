import math
import time
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from carla_env import CarlaEnv
import sys
import argparse


def main(
    model_name,
    load_model,
    town,
    fps,
    im_width,
    im_height,
    repeat_action,
    start_transform_type,
    sensors,
    enable_preview,
    steps_per_episode,
    seed=7,
    action_type="continuous",
):

    env = CarlaEnv(
        town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, action_type, enable_preview, steps_per_episode, playing=True
    )

    states = []
    actions = []
    try:
        model = SAC.load(load_model)
        obs = env.reset()
        total_reward = 0
        while True:
            # time.sleep(1 / 50.0)
            action, _states = model.predict(obs, deterministic=True)
            print(action)

            # action = [1]  # 纯MPC
            states.append(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print(total_reward)
                total_reward = 0
                obs = env.reset()
    finally:
        np.savetxt("states", states)
        np.savetxt("actions", actions)

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", help="name of model when saving")
    parser.add_argument("--load", type=str, default="sem_sac.zip", help="whether to load existing model")
    parser.add_argument("--map", type=str, default="Town04", help="name of carla map")
    parser.add_argument("--fps", type=int, default=20, help="fps of carla env")
    parser.add_argument("--width", type=int, help="width of camera observations")
    parser.add_argument("--height", type=int, help="height of camera observations")
    parser.add_argument("--repeat-action", type=int, default=1, help="number of steps to repeat each action")
    parser.add_argument("--start-location", type=str, help="start location type: [random, highway] for Town04")
    parser.add_argument("--sensor", action="append", type=str, help="type of sensor (can be multiple): [rgb, semantic]")
    parser.add_argument("--preview", action="store_true", help="whether to enable preview camera")
    parser.add_argument("--episode-length", type=int, help="maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=7, help="random seed for initialization")

    args = parser.parse_args()
    model_name = args.model_name
    load_model = args.load
    town = args.map
    fps = args.fps
    im_width = args.width
    im_height = args.height
    repeat_action = args.repeat_action
    start_transform_type = args.start_location
    sensors = args.sensor
    enable_preview = args.preview
    steps_per_episode = args.episode_length
    seed = args.seed

    main(
        model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, enable_preview, steps_per_episode, seed
    )
