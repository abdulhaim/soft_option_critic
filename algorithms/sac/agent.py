import itertools
import torch

from copy import deepcopy
from algorithms.sac.model import QFunction, Policy
from torch.optim import Adam
import torch.nn as nn


class SoftActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, args, tb_writer, log):
        super(SoftActorCritic, self).__init__()

        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.args = args
        self.tb_writer = tb_writer
        self.log = log

        # Q Function Definitions
        self.q_function_1 = QFunction(self.obs_dim, self.action_dim, self.args.hidden_dim)
        self.q_function_2 = QFunction(self.obs_dim, self.action_dim, self.args.hidden_dim)
        self.q_function_1_targ = deepcopy(self.q_function_1)
        self.q_function_2_targ = deepcopy(self.q_function_2)

        # Policy Definition
        self.policy = Policy(self.obs_dim, self.action_dim, self.args.hidden_dim, action_space.high[0])

        # Parameter Definitions
        self.q_params = itertools.chain(self.q_function_1.parameters(), self.q_function_1.parameters())
        self.q_params_target = itertools.chain(self.q_function_1_targ.parameters(), self.q_function_1_targ.parameters())
        self.policy_params = self.policy.parameters()

        self.pi_optimizer = Adam(self.policy_params, lr=self.args.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.args.lr)

        self.iteration = 0

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['state'], data['action'], data['reward'], data['next_state'], data['done']

        q1 = self.q_function_1(torch.cat([o, a], dim=-1))
        q2 = self.q_function_2(torch.cat([o, a], dim=-1))

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.policy(o2)

            # Target Q-values
            q1_pi_targ = self.q_function_1_targ(torch.cat([o2, a2], dim=-1))
            q2_pi_targ = self.q_function_2_targ(torch.cat([o2, a2], dim=-1))
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.args.gamma * (1 - d) * (q_pi_targ - self.args.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

        # Set up function for computing SAC pi loss

    def compute_loss_pi(self, data):
        o = data['state']
        pi, logp_pi = self.policy(o)
        q1_pi = self.q_function_1(torch.cat([o, pi], dim=-1))
        q2_pi = self.q_function_2(torch.cat([o, pi], dim=-1))
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.args.alpha * logp_pi - q_pi).mean()

        return loss_pi

    def update_loss_sac(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        self.tb_writer.log_data("q function loss", self.iteration, loss_q.item())

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        self.tb_writer.log_data("policy loss", self.iteration, loss_pi.item())

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.q_params, self.q_params_target):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

        self.iteration += 1

    def get_action(self, state):
        action, logp = self.policy(state)
        return action.detach().numpy(), logp.detach().numpy()