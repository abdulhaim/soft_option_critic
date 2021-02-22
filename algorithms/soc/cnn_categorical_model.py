import math
import torch
import torch.multiprocessing
from torch.distributions.categorical import Categorical

from torch import nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

class InterQFunction(torch.nn.Module):
    """
    There will only be one of these q functions
    Input: state, option
    Output: 1
    """

    def __init__(self, obs_dim, option_dim, hidden_size):
        super(InterQFunction, self).__init__()
        self.in_channels = 4
        self.input_shape = [4, 84, 84]

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, option_dim)
        )

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def forward(self, obs, gradient=False):
        if gradient:
            x = self.features(obs.permute(0, 3, 1, 2))
        else:
            x = self.features(obs.permute(2, 0, 1).unsqueeze(0))
        x = x.reshape(x.shape[0], self.feature_size())
        x = self.fc(x)
        return x



class IntraQFunction(torch.nn.Module):
    """
    There will be two q functions: q1 and q2
    Input: state, option, action
    Output: 1
    """

    def __init__(self, obs_dim, act_dim, option_dim, hidden_size):
        super(IntraQFunction, self).__init__()
        self.in_channels = 4
        self.input_shape = [4, 84, 84]
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size()+option_dim+1, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def forward(self, state, option, action, gradient=True):
        if gradient:
            features = self.features(state.permute(0, 3, 1, 2))
        else:
            features = self.features(state.permute(2, 0, 1).unsqueeze(0))
        features = Variable(features, requires_grad = True)
        features = torch.flatten(features, start_dim=1)
        features = Variable(features, requires_grad = True)
        x = torch.cat([features, option, action], dim=-1)
        q = self.fc(x)
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
        self.in_channels = 4
        self.input_shape = [4, 84, 84]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.option_dim = option_dim

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())

        self.last_w_layer = torch.randn((option_dim, self.feature_size(), act_dim))
        self.last_b_layer = torch.randn((option_dim, act_dim))

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def forward(self, obs, gradient=False, deterministic=False, with_logprob=True):
        if gradient:
            obs = obs.permute(0, 3, 1, 2)
        else:
            obs = obs.permute(2, 0, 1).unsqueeze(0)

        net_out = self.features(obs)
        x = torch.flatten(net_out, start_dim=1)
        mu = torch.matmul(x, self.last_w_layer)
        if gradient:
            b_mu = torch.unsqueeze(self.last_b_layer, axis=1)
        else:
            b_mu = self.last_b_layer
            mu = mu.reshape(mu.shape[0], mu.shape[2])

        mu = torch.add(mu, b_mu)
        action_probs = F.softmax(mu, dim=-1)
        action_distribution = Categorical(probs=action_probs)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = torch.argmax(mu, dim=-1)
        else:
            action = action_distribution.sample()

        if gradient:
            # Have to deal with situation of 0.0 probabilities because we can't do log 0
            z = action_probs == 0.0
            z = z.float() * 1e-8
            logs = torch.log(action_probs + z)
            logs = torch.gather(logs, 2, action.unsqueeze(-1))
        else:
            logs = None

        return action, logs

class BetaPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: 1
    """

    def __init__(self, obs_dim, option_dim, hidden_size):
        super(BetaPolicy, self).__init__()
        self.in_channels = 4
        self.input_shape = [4, 84, 84]
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())

        self.last_layer = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, option_dim))

    def feature_size(self):
        return self.net(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def forward(self, obs, gradient=False):
        if gradient:
            obs = obs.permute(0, 3, 1, 2)
        else:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        net_out = self.net(obs)
        net_out = net_out.reshape(net_out.shape[0], self.feature_size())
        x = self.last_layer(net_out)
        output = torch.sigmoid(x).squeeze(0)
        return output


class SOCModelCategorical(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, option_dim):
        super(SOCModelCategorical, self).__init__()
        # Inter-Q Function Definition

        self.inter_q_function_1 = InterQFunction(obs_dim, option_dim, hidden_size)
        self.inter_q_function_2 = InterQFunction(obs_dim, option_dim, hidden_size)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)
        self.intra_q_function_2 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)

        # Policy Definitions
        self.intra_option_policy = IntraOptionPolicy(obs_dim, act_dim, option_dim, hidden_size)
        self.beta_policy = BetaPolicy(obs_dim, option_dim, hidden_size)
