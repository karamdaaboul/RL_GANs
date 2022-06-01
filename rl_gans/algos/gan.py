import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

class GAN(object):
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.generator_update_freq     = args.generator_update_freq 
        self.discriminator_update_freq = args.discriminator_update_freq
        self.image_size = args.agent_image_size
        self.log_interval = args.log_interval
        
        # optimizers
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=args.discriminator_lr, betas=(args.discriminator_beta, 0.999))

        self.generator_optimizer = torch.optim.Adam(
            self.model.generator.parameters(), lr=args.generator_lr, betas=(args.generator_beta, 0.999))

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.discriminator.train(training)
        self.model.generator.train(training)


    def update_discriminator(self, obs, batch_size, L, step):
        criterion = nn.BCEWithLogitsLoss()
        z_dim   = self.model.generator.z_dim
        fake_noise = get_noise(batch_size, z_dim, device=self.device)
        fake = self.model.generator(fake_noise)
        discriminator_fake_pred = self.model.discriminator(fake.detach())
        discriminator_fake_loss = criterion(discriminator_fake_pred, torch.zeros_like(discriminator_fake_pred))
        discriminator_real_pred = self.model.discriminator(obs)
        discriminator_real_loss = criterion(discriminator_real_pred, torch.ones_like(discriminator_real_pred))
        discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) / 2

        # Keep track of the average discriminator loss
        # mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        if step % self.log_interval == 0:
            L.log('train_discriminator/loss', discriminator_loss, step)

        # Optimize the critic
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)        
        self.discriminator_optimizer.step()

    def update_generator(self, batch_size, L, step):
        # detach encoder, so we don't update it with the actor loss
        criterion = nn.BCEWithLogitsLoss()
        z_dim   = self.model.generator.z_dim
        fake_noise              = get_noise(batch_size, z_dim, device=self.device)
        fake                    = self.model.generator(fake_noise)
        discriminator_fake_pred = self.model.discriminator(fake)
        generator_loss          = criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))

        # Keep track of the average generator loss
        #mean_generator_loss += gen_loss.item() / display_step
        if step % self.log_interval == 0:
            L.log('train_generator/loss',generator_loss, step)

        # Optimize the generator
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()


    def update(self, replay_buffer, L, step):
        obs, _, _, _, _ = replay_buffer.sample()
        batch_size = len(obs)
        #if step % self.log_interval == 0:
        #    L.log('train/batch_reward', reward.mean(), step)

        if step % self.discriminator_update_freq == 0:
            self.update_discriminator(obs,batch_size, L, step)

        if step % self.generator_update_freq == 0:
            self.update_generator(batch_size, L, step)


    def save_model(self, dir, step):
        torch.save(self.model.state_dict(), os.path.join(dir, f'{step}.pt'))