import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .misc import *

OUT_DIM = {2: 39, 4: 35, 6: 31}
LOG_FREQ = 1000


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)

        self.out_dim = get_out_shape(obs_shape, self.layers)[-1]
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x/255.)


class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh()
        )
        self.out_dim = out_dim
        self.apply(weight_init)
    
    def forward(self, x):
        return self.projection(x)



class Encoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, cnn, projection):
        super().__init__()
        self.cnn = cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(self, encoder, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        if self.encoder  is not None:
            # Image as input
            input_dim = self.encoder.out_dim
        else:
            # state as input
            input_dim = obs_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
        if self.encoder is not None:
            x = self.encoder(x, detach=detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std
    

class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.mlp(obs_action)


class Critic(nn.Module):
    def __init__(self, encoder,obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.encoder = encoder
        if self.encoder  is not None:
            # Image as input
            input_dim = self.encoder.out_dim
        else:
            # state as input
            input_dim = obs_dim
        self.Q1 = QFunction(self.encoder.out_dim, action_dim, hidden_dim)
        self.Q2 = QFunction(self.encoder.out_dim, action_dim, hidden_dim)
        self.apply(weight_init)

    def forward(self, x, action, detach=False):
        if self.encoder  is not None:
            x = self.encoder(x, detach=detach)
        return self.Q1(x, action), self.Q2(x, action)

class Decoder(nn.Module):
    def __init__(self, num_channels, feature_dim, num_layers = 4, num_filters = 32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
        self.deconvs.append(nn.ConvTranspose2d(num_filters, num_channels, 3, stride=2, output_padding=1))


    def forward(self, h):
        h = torch.relu(self.fc(h))
        x = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        
        for i in range(0, self.num_layers - 1):
            x = torch.relu(self.deconvs[i](x))

        obs = self.deconvs[-1](x)
        return obs


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def recon(self, x):
        h = self.encoder(x)
        recon_x = self.decoder(h)
        return recon_x


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        num_channels: the number of channels of the output image, a scalar
        hidden_dim: the inner dimension, a scalar = num_filters
    '''
    def __init__(self, num_channels, z_dim=10, num_layers = 4, num_filters = 32):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]
        # Build the neural network
        self.layers = [self.make_gen_block(z_dim, num_filters, 3, stride=2)]
        for _ in range(1, num_layers-1):
            self.make_gen_block(num_filters , num_filters , kernel_size=4, stride=1)
        self.make_gen_block(num_filters, num_channels, kernel_size=4, final_layer=True)
        self.gen = nn.Sequential(*self.layers)

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:# Build the neural block
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        num_channels: the number of channels of the output image, a scalar
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, num_channels=3, num_filters=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(num_channels, num_filters),
            self.make_disc_block(num_filters, num_filters * 2),
            self.make_disc_block(num_filters * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    '''
    Function for completing a forward pass of the discriminator: Given an image tensor, 
    returns a 1-dimension tensor representing fake/real.
    Parameters:
        image: a flattened image tensor with dimension (im_dim)
    '''
    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

