import datetime
import gym
import numpy as np
import uuid
import torch

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()

to_np = lambda x: x.detach().cpu().numpy()

class ExcavationOrbit_Wrapper:
    def __init__(self, env) -> None:
        # Utils
        self.env = env
        self.num_envs = env.num_envs
        # Append Create list of unique indices
        self.unique_indices = []
        for i in range(env.unwrapped.num_envs):
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            id = f"{timestamp}-{str(uuid.uuid4().hex)}"
            self.unique_indices.append(id)
        self.device = env.device
        self.reset()

    def step(self, action_dict):
        # Action input is a dictionnary
        action_np = np.stack([entry['action'] for entry in action_dict])
        action_torch = torch.tensor(action_np, dtype=torch.float32, device=self.env.unwrapped.device)
        assert action_torch.shape[0] == self.num_envs

        # Orbit Env Stepping witohut reset if env terminated
        obs_orbit, rew, terminated, truncated, extras = self.env.step(action_torch)
        # compute dones 
        dones = (terminated | truncated)
        # this is only needed for infinite horizon tasks
        if not self.env.cfg.is_finite_horizon:
            extras["time_outs"] = to_np(truncated)
        # Convert obs, rew to the right format
        obs_dreamer = [
            {obs_key: obs_orbit[obs_key][env_idx].cpu().numpy() for obs_key in obs_orbit} # only key is policy anyway
            for env_idx in range(self.env.unwrapped.num_envs)
        ]

        reset_env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Complete ibs with is first and is terminal
        for i in range(self.env.num_envs):
            obs_dreamer[i]['is_first'] = False
            # If terminated: is_terminated = True
            if i in reset_env_ids:
                obs_dreamer[i]['is_terminal'] = True
            else:
                obs_dreamer[i]['is_terminal'] = False

        rew_np =  to_np(rew)
        done_np = to_np(dones)


        return obs_dreamer, rew_np, done_np, extras
    
    def reset_idx(self, indices):
        reset_env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Check that no bullshiet is done
        #assert (indices == to_np(reset_env_ids)).all()

        # Reset terminated envs
        if len(reset_env_ids) > 0:
            self.env.unwrapped.reset_idx(reset_env_ids)

        # Update derived measurements for resetted envs
        self.env.update_derived_measurements(reset_env_ids)
        # Update the obs at first, gives full set of obs
        obs_orbit = self.env.unwrapped.observation_manager.compute()
        self.env.last_actions[reset_env_ids] = 0


        # Transform observation in Orbit Format
        obs_dreamer = [
            {obs_key:  obs_orbit[obs_key][env_idx].cpu().numpy() for obs_key in obs_orbit} # only key is policy anyway
            for env_idx in range(self.env.num_envs)
        ]

        # Complete the obs with is_first and is_terminal
        for i in range(self.env.num_envs): # 
            # Cache Resetted state and ger
            if i in indices:
                # Adapt obs of new resetted env
                obs_dreamer[i]['is_first'] = True
                obs_dreamer[i]['is_terminal'] = False
                # Give the resetted env a new unique index
                timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                id = f"{timestamp}-{str(uuid.uuid4().hex)}"
                self.unique_indices[i] = id
            else:
                # Adapt obs for not resetted env since is_first and is_terminal have been remobved in envs.unwrapped.observation_manager.compute()
                obs_dreamer[i]['is_first'] = False
                obs_dreamer[i]['is_terminal'] = False
        
        return obs_dreamer
    
    
    
    def reset(self):
        obs_orbit, extras = self.env.reset()

        # Transform observation in Orbit Format
        obs_dreamer = [
            {obs_key:  obs_orbit[obs_key][env_idx].cpu().numpy() for obs_key in  obs_orbit} # only key is policy anyway
            for env_idx in range(self.env.num_envs)
        ]
        
        # Complete the obs with is_first and is_terminal
        for i in range(self.env.num_envs): # 
            # Cache Resetted state and ger
            obs_dreamer[i]['is_first'] = True
            obs_dreamer[i]['is_terminal'] = False
            # Give the resetted env a new unique index
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            id = f"{timestamp}-{str(uuid.uuid4().hex)}"
            self.unique_indices[i] = id

        return obs_dreamer
    
    def __len__(self):
        # Return the number of environments in the wrapper
        return (self.num_envs)