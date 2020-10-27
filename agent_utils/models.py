import torch
from torch import nn as nn
import torch.multiprocessing
import numpy as np
import math
import torch.nn.functional as F
from torch.distributions.normal import Normal


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class QFunction(torch.nn.Module):
    """
    There will be two of these q-functions
    Input: state, option
    Output: 1
    """

    def __init__(self, num_state, num_actions, hidden_size):
        super(QFunction, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state + num_actions, hidden_size)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
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
        return x


class InterQFunction(torch.nn.Module):
    """
    There will only be two of these q functions
    Input: state, option
    Output: 1
    """

    def __init__(self, num_state, num_options, hidden_size):
        super(InterQFunction, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state + num_options, hidden_size)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
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
        return x


class IntraQFunction(torch.nn.Module):
    """
    There will be two q functions: q1 and q2
    Input: state, option, action
    Output: 1
    """

    def __init__(self, num_state, num_actions, num_options, hidden_size):
        super(IntraQFunction, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state + num_options + num_actions, hidden_size)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
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
        return x


class Policy(torch.nn.Module):
    """
    There will be one inter-option policy
    Input: state
    Output: number of actions
    """

    def __init__(self, num_state, num_actions, hidden_size):
        super(Policy, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state, hidden_size)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3_mu = nn.Linear(hidden_size, num_actions)
        self.layer3_std = nn.Linear(hidden_size, num_actions)
        self.min_log_std = math.log(1e-6)

        self.module_list += [self.layer3_mu]
        self.module_list += [self.layer3_std]

        self.num_actions = num_actions
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        mu = self.layer3_mu(x)
        std = self.layer3_std(x)
        scale = torch.exp(torch.clamp(std, min=self.min_log_std))

        pi_distribution = Normal(loc=mu, scale=scale)
        pi_action = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)

        pi_action = torch.tanh(pi_action)
        pi_action = self.num_actions * pi_action

        return pi_action, logp_pi


class InterOptionPolicy(torch.nn.Module):
    """
    There will be one inter-option policy
    Input: state
    Output: number of actions
    """

    def __init__(self, num_state, num_options, hidden_size):
        super(InterOptionPolicy, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state, hidden_size)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3_mu = nn.Linear(hidden_size, num_options)
        self.layer3_std = nn.Linear(hidden_size, num_options)
        self.min_log_std = math.log(1e-6)

        self.module_list += [self.layer3_mu]
        self.module_list += [self.layer3_std]

        self.num_options = num_options
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        mu = self.layer3_mu(x)
        std = self.layer3_std(x)
        scale = torch.exp(torch.clamp(std, min=self.min_log_std))

        pi_distribution = Normal(loc=mu, scale=scale)
        pi_option = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_option).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_option - F.softplus(-2 * pi_option))).sum(axis=-1)

        pi_option = torch.tanh(pi_option)
        pi_option = self.num_options * pi_option

        return pi_option, logp_pi


class IntraOptionPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: number of actions
    """

    def __init__(self, num_state, num_option, num_actions, hidden_size):
        super(IntraOptionPolicy, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state + num_option, hidden_size)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3_mu = nn.Linear(hidden_size, num_actions)
        self.layer3_std = nn.Linear(hidden_size, num_actions)
        self.min_log_std = math.log(1e-6)

        self.module_list += [self.layer3_mu]
        self.module_list += [self.layer3_std]
        self.num_actions = num_actions
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        mu = self.layer3_mu(x)
        std = self.layer3_std(x)
        scale = torch.exp(torch.clamp(std, min=self.min_log_std))

        pi_distribution = Normal(loc=mu, scale=scale)
        pi_action = pi_distribution.rsample()

        logp_pi = pi_distribution.log_prob(pi_action).sum()
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)

        pi_action = torch.tanh(pi_action)
        pi_action = self.num_actions * pi_action

        return pi_action, logp_pi


class BetaPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: 1
    """

    def __init__(self, num_state, hidden_size):
        super(BetaPolicy, self).__init__()
        self.module_list = nn.ModuleList()
        self.layer1 = nn.Linear(num_state, hidden_size)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
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
