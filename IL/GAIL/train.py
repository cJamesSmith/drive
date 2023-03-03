import os
import json
import pickle

import torch
import gym

from models.gail import GAIL

from stable_baselines3 import SAC


def main(env_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    discrete = False
    action_dim = env.action_space.shape[0]
    device = "cuda"

    expert = SAC.load("./sac-expert-pedelwalker.zip")

    model = GAIL(state_dim, action_dim, discrete, config).to(device)

    results = model.train(env, expert)

    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt"))
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt"))
    if hasattr(model, "d"):
        torch.save(model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt"))


if __name__ == "__main__":
    main("BipedalWalker-v3")
