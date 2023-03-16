import gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from carla_env import CarlaEnv
import sys
import argparse
import torch


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
        town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, action_type, enable_preview, steps_per_episode, playing=False
    )
    # test_env = CarlaEnv(
    #     town,
    #     fps,
    #     im_width,
    #     im_height,
    #     repeat_action,
    #     start_transform_type,
    #     sensors,
    #     action_type,
    #     enable_preview=True,
    #     steps_per_episode=steps_per_episode,
    #     playing=True,
    # )

    try:
        if load_model is not None:
            print("********************yes*************************")
            model = SAC.load(
                load_model,
                env,
                # action_noise=NormalActionNoise(mean=np.array([0.0]), sigma=np.array([0.01])),
                learning_starts=0,
                verbose=2,
                # force_reset=True,
                # train_freq=10,
            )
        else:
            model = SAC(
                MlpPolicy,
                env,
                verbose=2,
                learning_starts=1000,
                seed=seed,
                device="cuda",
                tensorboard_log="./sem_sac",
                policy_kwargs={
                    "net_arch": [128, 128],
                    "activation_fn": torch.nn.ReLU,
                    # "share_features_extractor": True,
                    # "features_extractor_class": MlpPolicy,
                    # "features_extractor_kwargs": {"net_arch": [64, 64]},
                },
                # train_freq=(5, "step")
                action_noise=NormalActionNoise(mean=np.array([0]), sigma=np.array([0.01])),
            )
        # print(model.__dict__)
        # assert 1 == 0
        # for id in range(50):
        #     model.learn(total_timesteps=100000, log_interval=4, tb_log_name=model_name, progress_bar=True)
        #     model.save(model_name + "_" + str(id))
        model.learn(total_timesteps=100000, log_interval=20, tb_log_name=model_name, progress_bar=True)
        model.save(model_name)
    finally:
        env.close()
        # test_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", help="name of model when saving")
    parser.add_argument("--load", type=str, help="whether to load existing model")
    parser.add_argument("--map", type=str, default="Town04", help="name of carla map")
    parser.add_argument("--fps", type=int, default=30, help="fps of carla env")
    parser.add_argument("--width", type=int, help="width of camera observations")
    parser.add_argument("--height", type=int, help="height of camera observations")
    parser.add_argument("--repeat-action", type=int, help="number of steps to repeat each action")
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
