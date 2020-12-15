import torch
import torch.multiprocessing

from torch.distributions.categorical import Categorical
from torch import nn as nn
import numpy as np
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Policy(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.last_layer = nn.Linear(hidden_size, act_dim)
        self.act_limit = act_dim

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        x = self.last_layer(net_out)
        x = F.softmax(x, dim=-1)
        pi_distribution = Categorical(probs=x)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = x
        else:
            pi_action = pi_distribution.sample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        else:
            logp_pi = None
        return pi_action, logp_pi

class QFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size):
        super(QFunction, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim+act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class SACModel(nn.Module):

    def __init__(self, observation_space, action_space, hidden_size):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = 3

        # build policy and value functions
        self.policy = Policy(obs_dim, act_dim, hidden_size)
        self.q_function_1 = QFunction(obs_dim, act_dim, hidden_size)
        self.q_function_2 = QFunction(obs_dim, act_dim, hidden_size)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, logprob = self.policy(obs, deterministic, False)
            return a.numpy(), logprob
