"""Motion-generator training utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from iail_sim_picking.motion_generator.motion_generator_network import CVAE


class MotionGenerator(nn.Module):
    def __init__(self, action_dim, latent_dim, device, vae_lr=5e-4):
        super().__init__()
        self.actor_vae = CVAE(action_dim, latent_dim, device).to(device)
        self.kl_w = 0.0001

        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr)
        self.action_dim = action_dim
        self.device = device

    def kl_loss(self, mu, log_var):
        free_bits = 0.5
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # shape: [batch_size, latent_dim]

        free_bits_tensor = torch.tensor(free_bits, device=kl_per_dim.device)
        kl_freebits = torch.maximum(kl_per_dim, free_bits_tensor)
        kl_loss = kl_freebits.sum(dim=1).view(-1)

        # kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        return kl_loss.mean()

    def train_once(self, state, action):
        self.actor_vae.train()
        recons_action, _, mu, log_var = self.actor_vae(state, action)
        recons_loss = F.mse_loss(recons_action, action)
        kl_loss = self.kl_loss(mu, log_var)
        actor_vae_loss = recons_loss + kl_loss * self.kl_w

        self.actorvae_optimizer.zero_grad()
        actor_vae_loss.backward()
        self.actorvae_optimizer.step()

        return recons_loss.item(), kl_loss.item()

    def save(self, filename, directory):
        checkpoint_path = Path(directory) / f"{filename}.pth"
        torch.save(self.actor_vae.state_dict(), checkpoint_path)
