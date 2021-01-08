import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.last_layer = nn.Linear(hidden_size, act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(obs, 0)
        assert len(obs.shape) == 2, "batch process!"
        net_out = self.net(obs)
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
        self.q = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, obs):
        q = self.q(obs)
        return q


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
            a, logprob = self.policy(obs, deterministic, False)
            return a.numpy()[0], logprob
