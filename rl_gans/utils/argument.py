from re import X
import numpy as np
import argparse
import os
import math
import sys
import random
import time
import json
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    ##### Common #####
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_encoder_tau', default=0.05, type=float) 
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # gan
    parser.add_argument('--generator_update_freq', default=1, type=int)
    parser.add_argument('--discriminator_update_freq',  default=2, type=int)
    parser.add_argument('--generator_lr', default=1e-4, type=float)
    parser.add_argument('--generator_beta', default=0.5, type=float)
    parser.add_argument('--discriminator_lr', default=1e-4, type=float)
    parser.add_argument('--discriminator_beta', default=0.5, type=float)
    ##### Algorithm-Specific Parameters
    parser.add_argument('--agent', default='gan', type=str, help='curl, sacae, sac_pixel, sac_state, gan')
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--log_interval', default=25, type=int)

    args = parser.parse_args(args=[])

    if args.agent in ['curl']:
        args.env_image_size = 100
        args.agent_image_size = 84
    elif args.agent in ['sacae', 'sac_pixel','gan']:
        args.env_image_size = 84,
        args.agent_image_size = 84

    elif args.agent in ['sac_state']:
        args.env_image_size = None
        args.agent_image_size = None
        args.critic_encoder_tau = 0
        args.encoder_feature_dim = 0
        args.num_layers = 0
        args.num_filters = 0

    return args


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image