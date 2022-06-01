import numpy as np
import torch
import torch.nn as nn
import kornia
import random

from rl_gans.memory import ReplayBufferAugmentedDataset, ReplayBufferDataset

class ReplayBuffer(object):
    def __init__(self, iter, obs_shape, device, image_size=84, image_pad=None):
        self.iter = iter
        self.device = device
        self.image_size = image_size
        self.image_pad = image_pad

        if image_pad is not None:
            self.aug_trans = nn.Sequential(
                nn.ReplicationPad2d(image_pad),
                kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
    
    def sample(self):
        (obs, action, reward, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        
        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        not_done = not_done.to(self.device)
        
        return obs, action, reward, next_obs, not_done
      

    def sample_aug(self):
        (obs, action, reward, next_obs,aug_state) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        
        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        aug_state = aug_state.float().to(self.device)
        not_done = not_done.to(self.device)
        
        return obs, action, reward, next_obs, not_done, aug_state
      
    
    def sample_curl(self):
        (obs, action, reward, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        
        pos = obs.clone()
        
        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        not_done = not_done.to(self.device)
        pos = pos.float().to(self.device)             
        
        obs = random_crop(obs, self.image_size)
        next_obs = random_crop(next_obs, self.image_size)
        pos = random_crop(pos, self.image_size)
        
        cpc_kwargs = dict(obs_anchor=obs, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obs, action, reward, next_obs, not_done, cpc_kwargs


# referred from https://github.com/nicklashansen/dmcontrol-generalization-benchmark
def random_crop(x, size=84):
	"""Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
	assert isinstance(x, torch.Tensor) and x.is_cuda, \
		'input must be CUDA tensor'
	
	n = x.shape[0]
	img_size = x.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		return x

	x = x.permute(0, 2, 3, 1)

	w1 = torch.LongTensor(n).random_(0, crop_max)
	h1 = torch.LongTensor(n).random_(0, crop_max)

	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[torch.arange(n), w1, h1]

	return cropped


def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_buffer(replay_dir,replay_type, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, obs_shape, device, image_size, image_pad):
    max_size_per_worker = max_size // max(1, num_workers)


    if replay_type == "augmented":
        iterable = ReplayBufferAugmentedDataset(replay_dir,
                                max_size_per_worker,
                                num_workers,
                                nstep,
                                discount,
                                fetch_every=1000,
                                save_snapshot=save_snapshot)
    
    else:
        iterable = ReplayBufferDataset(replay_dir,
                        max_size_per_worker,
                        num_workers,
                        nstep,
                        discount,
                        fetch_every=1000,
                        save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    buffer = ReplayBuffer(iter(loader), obs_shape, device, image_size, image_pad)
    
    return buffer