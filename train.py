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

from dreamer import Dreamer, make_dataset, count_steps, make_env, make_env_ExcavationOrbit


def main(config):
    dreamer_tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        dreamer_tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = dreamer_tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = dreamer_tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = dreamer_tools.load_episodes(directory, limit=1)
    

    if ORBIT:
        train_envs = make_env_ExcavationOrbit()
        eval_envs = None
        from gym.spaces import Box, Dict
        from collections import OrderedDict
        acts = Box(-2, 2, (train_envs.env.scene.articulations['robot'].num_joints,), 'float32')
        obs = Dict([('policy', Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32))])
        # Now
        #config.eval_episode_num = 0
    else:
        make = lambda mode, id: make_env(config, mode, id)
        train_envs = [make("train", i) for i in range(config.envs)]
        eval_envs = [make("eval", i) for i in range(config.envs)]
        if config.parallel:
            train_envs = [Parallel(env, "process") for env in train_envs]
            eval_envs = [Parallel(env, "process") for env in eval_envs]
        else:
            train_envs = [Damy(env) for env in train_envs]
            eval_envs = [Damy(env) for env in eval_envs]
        acts = train_envs[0].action_space
        obs = train_envs[0].observation_space
    
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = dreamer_tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = dreamer_tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        obs,
        acts,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        dreamer_tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            if ORBIT:
                print("Start evaluation.")
                train_envs.reset()
                eval_policy = functools.partial(agent, training=False)
                dreamer_tools.simulate(
                    eval_policy,
                    train_envs,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                train_envs.reset()
                print("Benchmarking.")
                Benchmark_current_policy(eval_policy, train_envs, )
                train_envs.reset()

            else:
                print("Start evaluation.")
                eval_policy = functools.partial(agent, training=False)
                dreamer_tools.simulate(
                    eval_policy,
                    eval_envs,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                if config.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = dreamer_tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": dreamer_tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

def Benchmark_current_policy(eval_policy, eval_envs, num_steps = 200):
    
    obs_dreamer = eval_envs.reset()
    done = np.full(eval_envs.num_envs, False)

    # Set up env buffers
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
        
        # Action Generation
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

        # Env Stepping
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
    main(parser.parse_args(remaining))
    #play_policy(parser.parse_args(remaining))
    #Benchmark_loaded_policy(parser.parse_args(remaining))
