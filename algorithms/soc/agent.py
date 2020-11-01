import random
import torch

import itertools
import numpy as np
import torch.nn as nn

from math import exp
from copy import deepcopy
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Bernoulli
from misc.torch_utils import tensor
from algorithms.soc.model import InterQFunction, IntraQFunction, IntraOptionPolicy, BetaPolicy


class SoftOptionCritic(nn.Module):
    def __init__(self, observation_space, action_space, args, tb_writer, log):
        super(SoftOptionCritic, self).__init__()

        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.option_num = args.option_num
        self.args = args
        self.tb_writer = tb_writer
        self.log = log

        # Inter-Q Function Definitions
        self.inter_q_function_1 = InterQFunction(self.obs_dim, self.option_num, args.hidden_dim)
        self.inter_q_function_2 = InterQFunction(self.obs_dim, self.option_num, args.hidden_dim)
        self.inter_q_function_1_targ = deepcopy(self.inter_q_function_1)
        self.inter_q_function_2_targ = deepcopy(self.inter_q_function_2)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(self.obs_dim, self.action_dim, self.option_num, args.hidden_dim)
        self.intra_q_function_2 = IntraQFunction(self.obs_dim, self.action_dim, self.option_num, args.hidden_dim)
        self.intra_q_function_1_targ = deepcopy(self.intra_q_function_1)
        self.intra_q_function_2_targ = deepcopy(self.intra_q_function_2)

        self.beta_list = []
        self.intra_option_policies = []
        self.intra_policy_params = []

        # Policy Definitions
        for option_index in range(self.option_num):
            self.beta_list.append(BetaPolicy(self.obs_dim, args.hidden_dim))
            self.intra_option_policies.append(
                IntraOptionPolicy(self.obs_dim, self.action_dim, args.hidden_dim, action_space.high[0]))

        # Parameter Definitions
        self.q_params_inter = itertools.chain(
            self.inter_q_function_1.parameters(),
            self.inter_q_function_1.parameters())
        self.q_params_inter_target = itertools.chain(
            self.inter_q_function_1_targ.parameters(),
            self.inter_q_function_2_targ.parameters())

        self.q_params_intra = itertools.chain(self.intra_q_function_1.parameters(),
                                              self.intra_q_function_1.parameters())
        self.q_params_intra_target = itertools.chain(self.intra_q_function_1_targ.parameters(),
                                                     self.intra_q_function_2_targ.parameters())

        self.intra_policy_params = itertools.chain(*[policy.parameters() for policy in self.intra_option_policies])
        self.beta_params = itertools.chain(*[policy.parameters() for policy in self.beta_list])

        self.inter_q_function_optim = Adam(self.q_params_inter, lr=args.lr)
        self.intra_q_function_optim = Adam(self.q_params_intra, lr=args.lr)
        self.intra_policy_optim = Adam(self.intra_policy_params, lr=args.lr)
        self.beta_optim = Adam(self.beta_params, lr=args.lr)

        self.iteration = 0
        self.total_steps = 0

    def get_epsilon(self):
        eps = self.args.eps_min + (self.args.eps_start - self.args.eps_min) * exp(-self.total_steps / self.args.eps_decay)
        self.tb_writer.log_data("epsilon", self.total_steps, eps)
        self.total_steps += 1
        return eps

    def get_option(self, q, eta):  # soft-epsilon strategy
        if random.random() > eta:
            return np.argmax(np.max(q.data.numpy()))
        else:
            return random.randint(0, self.option_num - 1)

    def predict_option_termination(self, state, option):
        termination = self.beta_list[option](state)
        option_termination = Bernoulli(termination).sample()
        return bool(option_termination.item())

    def get_action(self, option, state):
        action, logp = self.intra_option_policies[option](state)
        return action.detach().numpy(), logp.detach().numpy()

    def compute_loss(self, data):
        state, option, action, logp, beta_prob, reward, next_state, done = \
            data['state'], data['option'], data['action'], data['logp'], data['beta_prob'], data['reward'], data['next_state'], data['done']
        ################################################################
        # Computing Intra-Q Function Update
        option = option.flatten().numpy().astype(int)
        one_hot = np.zeros((option.size, self.args.option_num))
        rows = np.arange(len(option))
        one_hot[rows, option] = 1

        q1_intra = self.intra_q_function_1(torch.cat([state, action, tensor(one_hot)], dim=-1))
        q2_intra = self.intra_q_function_2(torch.cat([state, action, tensor(one_hot)], dim=-1))

        q1_inter_all = self.inter_q_function_1(state)
        q2_inter_all = self.inter_q_function_2(state)

        option_indices = torch.LongTensor(option)
        option_indices = option_indices.unsqueeze(-1)
        q1_inter = torch.gather(q1_inter_all, 1, option_indices, out=None, sparse_grad=False)
        q2_inter = torch.gather(q2_inter_all, 1, option_indices, out=None, sparse_grad=False)

        with torch.no_grad():
            q1_inter_targ = self.inter_q_function_1_targ(next_state)
            q2_inter_targ = self.inter_q_function_2_targ(next_state)
            q_inter_targ = torch.min(q1_inter_targ, q2_inter_targ)
            q_inter_targ_current_option = torch.gather(q_inter_targ, 1, option_indices, out=None, sparse_grad=False)
            q_inter_targ_next_option = np.argmax(np.max(q_inter_targ.data.numpy()))

            # Computing Q-losses
            backup_intra = reward + self.args.gamma * (1 - done) * (((1 - beta_prob) * q_inter_targ_current_option) + (
                    beta_prob * q_inter_targ_next_option))

            ################################################################
            # Computing Beta Policy Loss
            q1_pi = self.inter_q_function_1(next_state)
            q2_pi = self.inter_q_function_2(next_state)
            q_pi = torch.min(q1_pi, q2_pi)

            q_pi_current_option = torch.gather(q_pi, 1, option_indices, out=None, sparse_grad=False)
            q_pi_next_option = np.argmax(np.max(q_pi.data.numpy()))
            advantage = q_pi_current_option - q_pi_next_option

            ################################################################
            # Computing Inter-Q Function Loss
            q1_intra_targ = self.intra_q_function_1_targ(torch.cat([state, action, tensor(one_hot)], dim=-1))
            q2_intra_targ = self.intra_q_function_2_targ(torch.cat([state, action, tensor(one_hot)], dim=-1))
            backup_inter = torch.min(q1_intra_targ, q2_intra_targ) - self.args.alpha * logp.view(100, 1)
            ################################################################

        # Intra-Q Function Loss
        loss_intra_q1 = ((q1_intra - backup_intra) ** 2).mean()
        loss_intra_q2 = ((q2_intra - backup_intra) ** 2).mean()
        loss_intra_q = loss_intra_q1 + loss_intra_q2

        # Beta Policy Loss
        loss_beta = (Variable(beta_prob, requires_grad=True) * Variable(advantage, requires_grad=True)).mean()

        # Inter-Q Function Loss
        loss_inter_q1 = ((q1_inter - backup_inter) ** 2).mean()
        loss_inter_q2 = ((q2_inter - backup_inter) ** 2).mean()
        loss_inter_q = loss_inter_q1 + loss_inter_q2

        # Intra-Option Policy Loss
        q_pi = torch.min(q1_intra, q2_intra)
        loss_intra_pi = (self.args.alpha * logp - q_pi).mean()

        return loss_inter_q, loss_intra_q, loss_intra_pi, loss_beta

    def update_loss_soc(self, data):
        # Clear all for parameters
        self.inter_q_function_optim.zero_grad()
        self.intra_q_function_optim.zero_grad()
        self.intra_policy_optim.zero_grad()
        self.beta_optim.zero_grad()

        # Compute losses
        loss_inter_q, loss_intra_q, loss_intra_pi, loss_beta = self.compute_loss(data)

        # Updating Inter-Q Functions
        loss_inter_q.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.q_params_inter, self.args.max_grad_clip)
        self.inter_q_function_optim.step()
        self.tb_writer.log_data("inter_q_function_loss", self.iteration, loss_inter_q.item())

        # Finally, update target inter-q networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.q_params_inter, self.q_params_inter_target):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

        #######################################################################################################

        # Updating Intra-Q Functions
        loss_intra_q.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.q_params_intra, self.args.max_grad_clip)
        self.intra_q_function_optim.step()
        self.tb_writer.log_data("intra_q_function_loss", self.iteration, loss_intra_q.item())

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params_intra:
            p.requires_grad = False

        # Updating Intra-Policy
        loss_intra_pi.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.q_params_intra, self.args.max_grad_clip)
        self.intra_policy_optim.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params_intra:
            p.requires_grad = True
        self.tb_writer.log_data("intra_q_policy_loss", self.iteration, loss_intra_pi.item())

        # Updating Beta-Policy
        loss_beta.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.beta_params, self.args.max_grad_clip)
        self.beta_optim.step()
        self.tb_writer.log_data("beta_policy_loss", self.iteration, loss_beta.item())

        # Finally, update target intra-q networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.q_params_intra, self.q_params_intra_target):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

        self.iteration += 1
