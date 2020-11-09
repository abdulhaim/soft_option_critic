import itertools
import torch

from copy import deepcopy
from algorithms.sac.model import SACModel
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
        self.model = SACModel(self.obs_dim, self.action_dim, self.args.hidden_dim, action_space.high[0])
        self.model_target = deepcopy(self.model)

        # Parameter Definitions
        self.q_params = itertools.chain(self.model.q_function_1.parameters(), self.model.q_function_2.parameters())

        self.pi_optimizer = Adam(self.model.policy.parameters(), lr=self.args.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.args.lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.model_target.parameters():
            p.requires_grad = False

        self.test_iteration = 0
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['state'], data['action'], data['reward'], data['next_state'], data['done']

        q1 = self.model.q_function_1(torch.cat([o, a], dim=-1))
        q2 = self.model.q_function_2(torch.cat([o, a], dim=-1))

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.policy(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q_function_1(torch.cat([o2, a2], dim=-1))
            q2_pi_targ = self.model_target.q_function_2(torch.cat([o2, a2], dim=-1))
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.args.gamma * (1 - d) * (q_pi_targ - self.args.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        o = data['state']
        pi, logp_pi = self.model.policy(o)
        q1_pi = self.model.q_function_1(torch.cat([o, pi], dim=-1))
        q2_pi = self.model.q_function_2(torch.cat([o, pi], dim=-1))
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.args.alpha * logp_pi - q_pi).mean()

        return loss_pi

    def update_loss_sac(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params, self.args.max_grad_clip)
        self.q_optimizer.step()
        self.tb_writer.log_data("q_function_loss", self.iteration, loss_q.item())

        # Next run one gradient descent step for pi
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.args.max_grad_clip)
        self.pi_optimizer.step()
        self.tb_writer.log_data("policy_loss", self.iteration, loss_pi.item())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.q_function_1.parameters(), self.model_target.q_function_1.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

            for p, p_targ in zip(self.model.q_function_2.parameters(), self.model_target.q_function_2.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

    def get_action(self, state):
        action, logp = self.model.policy(state)
        return action.detach().numpy(), logp.detach().numpy()