
import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).parent))

from pprint import pprint

import exploration as expl
import models
import dreamer_tools
from dreamer_tools import ORBIT
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
from collections import deque

to_np = lambda x: x.detach().cpu().numpy()

#--- Orbit
from omni.isaac.orbit.app import AppLauncher
from datetime import datetime
import carb
from datetime import datetime

# local imports
import cli_args  # i
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.num_envs = 10
args_cli.task= 'Isaac-m545-v0'
if ORBIT:
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from omni.isaac.orbit.envs import RLTaskEnvCfg
    import omni.isaac.contrib_tasks  # noqa: F401
    import omni.isaac.orbit_tasks  # noqa: F401
    from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg

from dreamer import Dreamer, make_dataset, count_steps, make_env, make_env_ExcavationOrbit, multipage


# VERY WRONG
def play_policy(config):
    from gym.spaces import Box, Dict

    eval_envs = make_env_ExcavationOrbit()
    acts = Box(-2, 2, (eval_envs.env.scene.articulations['robot'].num_joints,), 'float32')
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    obs = Dict([('policy', Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32))])
    train_dataset = None
    logdir = "./logdir/dmc_walker_walk"
    logger = dreamer_tools.Logger(logdir, config.action_repeat * 0)
    agent = Dreamer(
        obs,
        acts,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    
    from pathlib import Path
    logdir = Path(logdir)
    # Load the Policy
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        dreamer_tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    
    # Define the evaluation policy, making sure the agent is in evaluation mode (no training).
    eval_policy = functools.partial(agent, training=False)
    
    obs_dreamer = eval_envs.reset()
    done = np.full(eval_envs.num_envs, False)
    while True:
        with torch.no_grad():
            obs = {k: np.stack([o[k] for o in obs_dreamer]) for k in obs_dreamer[0] if "log_" not in k}
            action, _ = eval_policy(obs, done, training=False)

        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(eval_envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(eval_envs)
        obs_dreamer, reward, done, info = eval_envs.step(action)
        # Put back the obs in orbit format since we just want to feed the agent and not log
        if done.any():
            pprint(info["episode_pos_term_counts"])
            pprint(info["episode_neg_term_counts"])
            indices = np.nonzero(done)[0]
            obs_dreamer = eval_envs.reset_idx(indices)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="dmc_proprio", help="Model configurations")
    parser.add_argument("--task", default="dmc_walker_walk", help="Task to perform")
    parser.add_argument("--logdir", default="./logdir/dmc_walker_walk", help="Directory for log output")

    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # Ensure args.configs is a list, even if it contains only one element
    name_list = ["defaults", args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = dreamer_tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    play_policy(parser.parse_args(remaining))