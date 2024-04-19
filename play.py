import argparse
import functools
import os
import pathlib
import sys

import dreamer_tools
from dreamer_tools import ORBIT
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
#--- Orbit
from omni.isaac.orbit.app import AppLauncher
from datetime import datetime
import carb
from datetime import datetime
from pprint import pprint
import numpy as np

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
    from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


def play_policy(agent, env, logdir):
    # Load the Policy
    if (logdir / "model_355000.pt").exists():
        checkpoint = torch.load(logdir / "model_355000.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        dreamer_tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        # Ensure the agent does not pretrain again if it has already completed pretraining.
        agent._should_pretrain._once = False
    obs = env.reset()
    done = np.full(env.num_envs, True)
    obs['is_first'] = np.full(env.num_envs, True)
    obs['is_terminal'] = np.full(env.num_envs, False)
    while True:
        with torch.no_grad():
            action, _ = agent(obs, done, training=False)
        obs_dreamer, reward, done, info = env.step(action)
        # Put back the obs in orbit format since we just want to feed the agent and not log
        indices = []
        if done.any():
            pprint(info["normalized_neg_term_sums"])
            pprint(info["normalized_pos_term_sums"])
            indices = np.nonzero(done)[0]
            obs_dreamer = env.reset_idx(indices)
        obs = {k: np.stack([o[k] for o in obs_dreamer]) for k in obs_dreamer[0] if "log_" not in k}
        obs['is_first'] = np.full(env.num_envs, False)
        obs['is_terminal'] = np.full(env.num_envs, False)
        for i in indices:
            obs['is_first'][i] = True