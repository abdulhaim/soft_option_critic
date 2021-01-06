import math
import torch
import torch.multiprocessing
from torch.distributions.categorical import Categorical

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

    def forward(self, obs):
        q = self.q(obs)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


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

    def forward(self, state, option, action):
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

    def __init__(self, obs_dim, act_dim, option_dim, hidden_size):
        super(IntraOptionPolicy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.last_w_layer = torch.randn((option_dim, hidden_size, act_dim))
        self.last_b_layer = torch.randn((option_dim, act_dim))

    def forward(self, obs, gradient=False, deterministic=False, with_logprob=True):
        x = self.net(obs)
        mu = torch.matmul(x, self.last_w_layer)

        if gradient:
            b_mu = torch.unsqueeze(self.last_b_layer, axis=1)
        else:
            b_mu = self.last_b_layer

        mu = torch.add(mu, b_mu)

        action_probs = F.softmax(mu, dim=-1)
        action_distribution = Categorical(probs=action_probs)

        if deterministic:
            # Only used for evaluating policy at test time.
            action = x
        else:
            action = action_distribution.sample().cpu()

        if with_logprob:
            # Have to deal with situation of 0.0 probabilities because we can't do log 0
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_action_probabilities = torch.log(action_probs + z)
            # logs = torch.gather(log_action_probabilities, 1, action.unsqueeze(-1))
        else:
            logs = None

        return action, log_action_probabilities

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

    def forward(self, inputs):
        x = self.net(inputs)
        return torch.sigmoid(x)


class SOCModelCategorical(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, option_dim):
        super(SOCModelCategorical, self).__init__()
        # Inter-Q Function Definitions
        self.inter_q_function_1 = InterQFunction(obs_dim, option_dim, hidden_size)
        self.inter_q_function_2 = InterQFunction(obs_dim, option_dim, hidden_size)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)
        self.intra_q_function_2 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)

        # Policy Definitions
        self.intra_option_policy = IntraOptionPolicy(obs_dim, act_dim, option_dim, hidden_size)
        self.beta_policy = BetaPolicy(obs_dim, option_dim, hidden_size)
