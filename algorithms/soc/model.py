import math
import torch
import torch.multiprocessing

from misc.torch_utils import weights_init, norm_col_init
from torch.distributions.normal import Normal
from torch import nn as nn
import numpy as np
import torch.nn.functional as F


class InterQFunction(torch.nn.Module):
    """
    There will only be one of these q functions
    Input: state, option
    Output: 1
    """

    def __init__(self, obs_dim, option_dim, hidden_size):
        super(InterQFunction, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, option_dim)
        )

    def forward(self, obs, gradient=True):
        q = self.q(obs)
        return torch.squeeze(q,-1)  # Critical to ensure q has right shape.


class IntraQFunction(torch.nn.Module):
    """
    There will be two q functions: q1 and q2
    Input: state, option, action
    Output: 1
    """

    def __init__(self, obs_dim, act_dim, option_dim, hidden_size):
        super(IntraQFunction, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + option_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, option, action, gradient=True):
        q = self.q(torch.cat([state, option, action], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class IntraOptionPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: number of actions
    """

    def __init__(self, obs_dim, act_dim, option_dim, hidden_size, act_limit):
        super(IntraOptionPolicy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.w_mu_layer = torch.randn((option_dim, hidden_size, act_dim))
        self.w_log_std_layer = torch.randn((option_dim, hidden_size, act_dim))

        self.b_mu_layer = torch.randn((option_dim, act_dim))
        self.b_log_std_layer = torch.randn((option_dim, act_dim))
        self.act_limit = act_limit

    def forward(self, obs, gradient=False, deterministic=False, with_logprob=True):
        x = self.net(obs)
        mu = torch.matmul(x, self.w_mu_layer)
        log_std = torch.matmul(x, self.w_log_std_layer)

        if gradient:
            b_log_std = torch.unsqueeze(self.b_log_std_layer, axis=1)
            b_mu = torch.unsqueeze(self.b_mu_layer, axis=1)
        else:
            b_log_std = self.b_log_std_layer
            b_mu = self.b_mu_layer

        mu = torch.add(mu, b_mu)
        log_std = torch.add(log_std, b_log_std)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(loc=mu, scale=std)
        pi_action = pi_distribution.rsample()  # NOTE Needed for reparameterization
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)

        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


class BetaPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: 1
    """

    def __init__(self, obs_dim, option_dim, hidden_size):
        super(BetaPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, option_dim))

    def forward(self, inputs, gradient=True):
        x = self.net(inputs)
        return torch.sigmoid(x)


class SOCModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, option_dim, act_limit):
        super(SOCModel, self).__init__()
        # Inter-Q Function Definitions
        self.inter_q_function_1 = InterQFunction(obs_dim, option_dim, hidden_size)
        self.inter_q_function_2 = InterQFunction(obs_dim, option_dim, hidden_size)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)
        self.intra_q_function_2 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)

        # Policy Definitions
        self.intra_option_policy = IntraOptionPolicy(obs_dim, act_dim, option_dim, hidden_size, act_limit)
        self.beta_policy = BetaPolicy(obs_dim, option_dim, hidden_size)
