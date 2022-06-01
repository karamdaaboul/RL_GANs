import numpy as np
from collections import defaultdict
import traceback

from rl_gans.memory.replay_buffer_base import ReplayBufferStorageBase, ReplayBufferDatasetBase, episode_len


class ReplayBufferStorage(ReplayBufferStorageBase):
    def __init__(self, replay_dir):
        super().__init__(replay_dir)
        self._current_episode = defaultdict(list)

    def add(self, state, action, reward, done):
        if state is not None:
            self._current_episode['s'].append(state)
        if action is not None:
            self._current_episode['a'].append(action)
        if reward is not None:
            self._current_episode['r'].append(reward)
        
        if done:
            episode = dict()
            episode['s']  = np.array(self._current_episode['s'], np.uint8)
            episode['a']  = np.array(self._current_episode['a'], np.float32)
            episode['r']  = np.array(self._current_episode['r'], np.float32)

            self._current_episode = defaultdict(list)
            self._store_episode(episode)


class ReplayBufferDataset(ReplayBufferDatasetBase):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                fetch_every, save_snapshot):
        super().__init__(replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot)

    def _sample(self):       
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        idx       = np.random.randint(0, episode_len(episode) - self._nstep + 1)
        obs       = episode['s'][idx]
        action    = episode['a'][idx]
        next_obs  = episode['s'][idx + self._nstep]
        reward    = np.zeros_like(episode['r'][idx])
        discount = 1
        for i in range(self._nstep):
            step_reward = episode['r'][idx]
            reward += discount * step_reward
            discount *= self._discount
        return (obs, action, reward, next_obs)