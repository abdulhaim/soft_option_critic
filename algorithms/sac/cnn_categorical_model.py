import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import nn as nn
import torch.autograd as autograd


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
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
            nn.Linear(512, act_dim)
        )

    def feature_size(self):
        return self.net(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def forward(self, obs, deterministic=False, with_logprob=True):
        if with_logprob:
            obs = obs.permute(0, 3, 1, 2)
        else:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(obs, 0)
        # assert len(obs.shape) == 4, "batch process!"
        net_out = self.net(obs)

        net_out = net_out.reshape(net_out.shape[0], self.feature_size())
        x = self.last_layer(net_out)

        if deterministic:
            # Only used for evaluating policy at test time.
            action = torch.argmax(x, dim=-1)
        else:
            action_probs = F.softmax(x, dim=-1)
            action_distribution = Categorical(probs=action_probs)
            action = action_distribution.sample().cpu()

        if with_logprob:
            # Have to deal with situation of 0.0 probabilities because we can't do log 0
            # log_action_probs = F.log_softmax(x, dim=-1)
            z = (action_probs == 0.0).float() * 1e-5
            log_action_probabilities = torch.log(action_probs + z)
            return action_probs, log_action_probabilities
        else:
            logs = None
            return action, logs


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(QFunction, self).__init__()
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
            nn.Linear(512, act_dim)
        )

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def forward(self, obs):
        x = self.features(obs.permute(0, 3, 1, 2))
        x = x.reshape(x.shape[0], self.feature_size())
        x = self.fc(x)
        return x


class SACModelCategorical(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.policy = Policy(obs_dim, act_dim, hidden_size)
        self.q_function_1 = QFunction(obs_dim, act_dim, hidden_size)
        self.q_function_2 = QFunction(obs_dim, act_dim, hidden_size)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, logprob = self.policy(obs, deterministic, with_logprob=False)
            return a.cpu().numpy()[0], logprob
