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

def Benchmark_loaded_policy(config):
    from gym.spaces import Box, Dict

    eval_envs = make_env_ExcavationOrbit()
    acts = Box(-2, 2, (eval_envs.env.scene.articulations['robot'].num_joints,), 'float32')
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    obs = Dict([('policy', Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32))])
    train_dataset = None
    logdir = "./logdir/dmc_walker_walk"
    #logger = dreamer_tools.Logger(logdir, config.action_repeat * 0)
    logger = dreamer_tools.WandbLogger(logdir, config, 0)
    agent = Dreamer(
        obs,
        acts,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
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
    obs = {k: np.stack([o[k] for o in obs_dreamer]) for k in obs_dreamer[0] if "log_" not in k}
    done = np.full(eval_envs.num_envs, False)

    # Set up env buffers
    num_steps = 500
    num_data = eval_envs.num_envs * num_steps
    bucket_aoa = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    bucket_vel = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    base_vel = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    max_depth = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    pullup_dist = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    in_soil = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    bucket_x = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    bucket_z = torch.zeros(eval_envs.num_envs, num_steps, device=eval_envs.device)
    # need to keep track of it manually, because step() resets if done but we only log after step()
    ep_lens = torch.zeros(eval_envs.num_envs, device=eval_envs.device)
    ep_len_counts = {}
    ep_len_counts["timeout"] = deque()
    for name in eval_envs.env.termination_excavation.neg_term_names:
        ep_len_counts["neg_" + name] = deque()
    ep_len_counts["close"] = deque()
    ep_len_counts["full"] = deque()


    episode_count = 0
    agent_state = None
    # 
    episode_count = 0
    step, episode = 0, 0
    done = np.ones((eval_envs.num_envs), bool)
    length = np.zeros((eval_envs.num_envs), np.int32)
    obs = [None] * eval_envs.num_envs
    agent_state = None
    reward = [0] * eval_envs.num_envs
    i = 0
    while (num_steps and i < num_steps):
        obs = {k: np.stack([o[k] for o in obs_dreamer]) for k in obs_dreamer[0] if "log_" not in k}
        action, agent_state = eval_policy(obs, done, agent_state)

        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(eval_envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(eval_envs)

        # Minimal logging
        bucket_aoa[:, i] = eval_envs.env.m545_measurements.bucket_aoa
        bucket_vel[:, i] = torch.linalg.norm(eval_envs.env.m545_measurements.bucket_vel_w, dim=-1)
        base_vel[:, i] = torch.linalg.norm(eval_envs.env.m545_measurements.root_lin_vel_w, dim=-1)
        bucket_z[:, i] = eval_envs.env.m545_measurements.bucket_pos_w[:, 2]
        max_depth[:, i] = eval_envs.env.soil.get_max_depth_height_at_pos(eval_envs.env.m545_measurements.bucket_pos_w[:, 0:1]).squeeze()
        pullup_dist[:, i] = eval_envs.env.pullup_dist
        bucket_x[:, i] = eval_envs.env.m545_measurements.bucket_pos_w[:, 0]
        in_soil[:, i] = (eval_envs.env.soil.get_bucket_depth() > 0.0).squeeze()

        obs_dreamer, reward, done, info = eval_envs.step(action)

        ep_lens += 1
        length += 1
        step += len(eval_envs)
        length *= 1 - done
        i += 1

        # Put back the obs in orbit format since we just want to feed the agent and not log
        if done.any():
            #pprint(info["episode_pos_term_counts"])
            #pprint(info["episode_neg_term_counts"])
            indices = np.nonzero(done)[0]
            obs_dreamer = eval_envs.reset_idx(indices)
            episode_count += len(indices)

            episode_count += len(indices)
            ep_len_counts["timeout"].extend(ep_lens[eval_envs.env.termination_excavation.time_out_buf].tolist())
            for name in eval_envs.env.termination_excavation.neg_term_names:
                ep_len_counts["neg_" + name].extend(ep_lens[eval_envs.env.termination_excavation.episode_neg_term_buf[name]].tolist())
            ep_len_counts["close"].extend(ep_lens[eval_envs.env.termination_excavation.close_pos_term_buf].tolist())
            ep_len_counts["full"].extend(ep_lens[eval_envs.env.termination_excavation.full_pos_term_buf].tolist())

        
    #-- Plotting
    # terminations, percentage [0,1]
    values = []
    labels = []
    for key, value in eval_envs.env.termination_excavation.episode_neg_term_counts.items():
        values.append(torch.sum(value).item() / episode_count)
        labels.append(key)

    sum_pos_term = 0
    for key, value in eval_envs.env.termination_excavation.episode_pos_term_counts.items():
        values.append(torch.sum(value).item() / episode_count)
        labels.append(key)
        sum_pos_term += int(torch.sum(value).item())

    full_term = torch.sum(eval_envs.env.termination_excavation.episode_pos_term_counts["desired_full"]).item() / episode_count
    close_term = torch.sum(eval_envs.env.termination_excavation.episode_pos_term_counts["desired_close"]).item() / episode_count


    values.append(torch.sum(eval_envs.env.termination_excavation.time_out_count).item() / episode_count)
    labels.append("timeout")

    _, ax = plt.subplots()
    ax.tick_params(axis="x", which="major", labelsize=6)
    ax.bar(np.arange(len(values)), values, tick_label=labels)
    ax.set_title(
        "close ({:.2f}) & full ({:.2f}) term/tot term: {} / {} [{:.2f}%]".format(
            close_term, full_term, sum_pos_term, episode_count, 100.0 * sum_pos_term / episode_count
        )
    )
    ax.grid()
    import statistics
    # stats violating negative termination conditions
    def log_and_print_stats(name, errs, num_data, error_dict):
        error_dict[name] = errs
        print(
            "{:<25} num/num_data: {:<10.2e} mean: {:<7.2f} std: {:<7.2f} min: {:<7.2f} max: {:<7.2f}".format(
                name,
                len(errs) / num_data,
                statistics.mean(errs) if len(errs) > 1 else np.nan,
                statistics.stdev(errs) if len(errs) > 1 else np.nan,
                min(errs) if len(errs) > 0 else np.nan,
                max(errs) if len(errs) > 0 else np.nan,
            )
        )


    print("num data samples: ", num_data)
    error_dict = {}

    # bucket aoa
    bad_aoa = bucket_aoa < 0.0
    fast_enough = bucket_vel > eval_envs.env.cfg.terminations_excavation.bucket_vel_aoa_threshold
    ids = torch.where(torch.logical_and(in_soil, torch.logical_and(bad_aoa, fast_enough)))
    errs = bucket_aoa[ids] - 0.0
    log_and_print_stats("bucket_aoa", errs.tolist(), num_data, error_dict)
    # bucket vel
    ids = torch.where(eval_envs.env.m545_measurements.bucket_vel_norm > eval_envs.env.cfg.terminations_excavation.max_bucket_vel)
    errs = eval_envs.env.m545_measurements.bucket_vel_norm[ids] - eval_envs.env.cfg.terminations_excavation.max_bucket_vel
    log_and_print_stats("bucket_vel", errs.tolist(), num_data, error_dict)

    # base vel
    ids = torch.where(base_vel > eval_envs.env.cfg.terminations_excavation.max_base_vel)
    errs = base_vel[ids] - eval_envs.env.cfg.terminations_excavation.max_base_vel
    log_and_print_stats("base_vel", errs.tolist(), num_data, error_dict)

    # max depth
    ids = torch.where(bucket_z < (max_depth - eval_envs.env.cfg.terminations_excavation.max_depth_overshoot))
    errs = bucket_z[ids] - (max_depth[ids] - eval_envs.env.cfg.terminations_excavation.max_depth_overshoot)
    log_and_print_stats("max_depth", errs.tolist(), num_data, error_dict)

    # pullup
    ids = torch.where(bucket_x < pullup_dist)
    errs = bucket_x[ids] - pullup_dist[ids]
    log_and_print_stats("pullup", errs.tolist(), num_data, error_dict)

    # episode lengths
    log_and_print_stats("len timeout", ep_len_counts["timeout"], num_data, error_dict)
    log_and_print_stats("len close", ep_len_counts["close"], num_data, error_dict)
    log_and_print_stats("len full", ep_len_counts["full"], num_data, error_dict)

    for name in eval_envs.env.termination_excavation.neg_term_names:
        log_and_print_stats("len neg_" + name, ep_len_counts["neg_" + name], num_data, error_dict)

    for key, value in error_dict.items():
        fig, ax = plt.subplots()
        ax.boxplot(value)
        ax.set_xticklabels([key], fontsize=6)
        quantiles = np.quantile(value, [0.25, 0.5, 0.75]) if len(value) > 1 else [0, 0, 0]
        ax.set_yticks(quantiles)
        ax.set_title(
            "q1-q3: {}, steps/total_steps: {} / {} [{:.2f}%]".format(
                " | ".join([str(np.round(x, 2)) for x in quantiles]), len(value), num_data, 100.0 * len(value) / num_data
            )
        )
        ax.grid()

    # save all plots in pdf and open
    filename = os.path.join("/home/jonas/Desktop", "stats.pdf")
    multipage(filename)
    os.system("xdg-open " + filename)
    plt.close("all")
        
    eval_envs.close()

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
    Benchmark_loaded_policy(parser.parse_args(remaining))