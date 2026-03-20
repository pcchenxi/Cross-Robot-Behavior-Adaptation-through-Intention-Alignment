"""
Network definitions for motion generation.
"""
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CVAE(nn.Module):
    def __init__(self, action_dim, latent_dim, device):
        super().__init__()
        image_channel = 3
        image_feature_dim = 800

        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channel, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            Flatten(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(image_feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mean = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)

        self.action_decoder = nn.Sequential(
            nn.Linear(image_feature_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        image_feature = self.image_encoder(state)
        combined_feature = self.action_encoder(torch.cat([image_feature, action], 1))

        mean = self.mean(combined_feature)
        log_var = self.log_var(combined_feature)
        std = torch.exp(log_var / 2)
        z = mean + std * torch.randn_like(std).float()

        action = self.action_decoder(torch.cat([image_feature, z], 1))
        return action, z, mean, log_var

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device)

        image_feature = self.image_encoder(state)
        action = self.action_decoder(torch.cat([image_feature, z], 1))
        return action
