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
    def __init__(self, num_state, num_options, hidden_dim):
        super(InterQFunction, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state, hidden_dim)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, num_options)
        self.module_list += [self.layer3]

        self.apply(weights_init)
        self.layer3.weight.data = norm_col_init(
            self.layer3.weight.data, 1.0)
        self.layer3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        x = self.layer3(x)
        return torch.squeeze(x, -1)


class IntraQFunction(torch.nn.Module):
    """
    There will be two q functions: q1 and q2
    Input: state, option, action
    Output: 1
    """
    def __init__(self, num_state, num_actions, num_options, hidden_dim):
        super(IntraQFunction, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state + num_options + num_actions, hidden_dim)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.module_list += [self.layer3]

        self.apply(weights_init)
        self.layer3.weight.data = norm_col_init(
            self.layer3.weight.data, 1.0)
        self.layer3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        x = self.layer3(x)
        return torch.squeeze(x, -1)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class IntraOptionPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: number of actions
    """

    def __init__(self, num_state, num_actions, hidden_dim, action_limit):
        super(IntraOptionPolicy, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state, hidden_dim)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3_mu = nn.Linear(hidden_dim, num_actions)
        self.layer3_std = nn.Linear(hidden_dim, num_actions)
        self.min_log_std = math.log(1e-6)
        self.action_limit = action_limit
        self.module_list += [self.layer3_mu]
        self.module_list += [self.layer3_std]
        self.num_actions = num_actions
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        mu = self.layer3_mu(x)
        log_std = self.layer3_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(loc=mu, scale=std)
        pi_action = pi_distribution.rsample()  # NOTE Needed for reparameterization
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        pi_action = torch.tanh(pi_action)
        pi_action = self.action_limit * pi_action

        return pi_action, logp_pi


class BetaPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: 1
    """

    def __init__(self, num_state, hidden_dim):
        super(BetaPolicy, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state, hidden_dim)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.module_list += [self.layer3]

        self.apply(weights_init)
        self.layer3.weight.data = norm_col_init(
            self.layer3.weight.data, 1.0)
        self.layer3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        x = self.layer3(x)
        return torch.sigmoid(x)

#
class SOCModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, option_num, act_limit):
        super(SOCModel, self).__init__()
        # Inter-Q Function Definitions
        self.inter_q_function_1 = InterQFunction(obs_dim, option_num, hidden_dim)
        self.inter_q_function_2 = InterQFunction(obs_dim, option_num, hidden_dim)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(obs_dim, action_dim, option_num, hidden_dim)
        self.intra_q_function_2 = IntraQFunction(obs_dim, action_dim, option_num, hidden_dim)

        self.beta_list = []
        self.intra_option_policies = []
        self.intra_policy_params = []

        # Policy Definitions
        for option_index in range(option_num):
            self.beta_list.append(BetaPolicy(obs_dim, hidden_dim))
            self.intra_option_policies.append(
                IntraOptionPolicy(obs_dim, action_dim, hidden_dim, act_limit))
