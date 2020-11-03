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
from misc.torch_utils import tensor, convert_onehot
from algorithms.soc.model import SOCModel


class SoftOptionCritic(nn.Module):
    def __init__(self, observation_space, action_space, args, tb_writer, log):
        super(SoftOptionCritic, self).__init__()

        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.option_num = args.option_num
        self.args = args
        self.tb_writer = tb_writer
        self.log = log

        self.model = SOCModel(self.obs_dim, self.action_dim, self.args.hidden_dim, self.option_num, action_space.high[0])
        self.model_target = deepcopy(self.model)

        # Parameter Definitions
        self.q_params_inter = itertools.chain(
            self.model.inter_q_function_1.parameters(),
            self.model.inter_q_function_2.parameters())

        self.q_params_intra = itertools.chain(
            self.model.intra_q_function_1.parameters(),
            self.model.intra_q_function_2.parameters())

        self.intra_policy_params = itertools.chain(*[policy.parameters() for policy in self.model.intra_option_policies])
        self.beta_params = itertools.chain(*[policy.parameters() for policy in self.model.beta_list])

        self.inter_q_function_optim = Adam(self.q_params_inter, lr=args.lr)
        self.intra_q_function_optim = Adam(self.q_params_intra, lr=args.lr)
        self.intra_policy_optim = Adam(self.intra_policy_params, lr=args.lr)
        self.beta_optim = Adam(self.beta_params, lr=args.lr)

        self.iteration = 0

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.model_target.parameters():
            p.requires_grad = False

    def get_epsilon(self, decay=False):
        if decay:
            eps = self.args.eps_min + (self.args.eps_start - self.args.eps_min) * exp(
                -self.iteration / self.args.eps_decay)
            self.tb_writer.log_data("epsilon", self.iteration, eps)
        else:
            eps = self.args.eps_min
        return eps

    def get_option(self, state, eta, gradient=False):  # soft-epsilon strategy
        q_function = torch.min(self.model.inter_q_function_1(state), self.model.inter_q_function_2(state))
        option = np.argmax(q_function.data.numpy(), axis=-1)
        if gradient:
            return option
        else:
            if random.random() < eta:
                option = random.randint(0, self.option_num - 1)
            self.tb_writer.log_data("option", self.iteration, option)
            return option

    def predict_option_termination(self, state, option):
        termination = self.model.beta_list[option](state)
        option_termination = Bernoulli(termination).sample()
        return termination, bool(option_termination.item())

    def get_action(self, option, state):
        action, logp = self.model.intra_option_policies[option](state)
        return action.detach().numpy(), logp.detach().numpy()

    def compute_loss_beta(self, next_state, option_indices, beta_prob):
        with torch.no_grad():
            # Computing Beta Policy Loss
            q1_pi = self.model.inter_q_function_1(next_state)
            q2_pi = self.model.inter_q_function_2(next_state)
            q_pi = torch.min(q1_pi, q2_pi)

            q_pi_current_option = torch.gather(q_pi, 1, option_indices)
            q_pi_next_option = np.max(q_pi.data.numpy())
            advantage = q_pi_current_option - q_pi_next_option

        # Beta Policy Loss
        loss_beta = (beta_prob * advantage.detach()).mean()
        return loss_beta

    def compute_loss_inter(self, state, option_indices, one_hot_option, current_actions, logp):
        q1_inter_all = self.model.inter_q_function_1(state)
        q2_inter_all = self.model.inter_q_function_2(state)

        q1_inter = torch.gather(q1_inter_all, 1, option_indices)
        q2_inter = torch.gather(q2_inter_all, 1, option_indices)

        with torch.no_grad():
            # Computing Inter-Q Function Loss
            q1_intra_targ = self.model_target.intra_q_function_1(
                torch.cat([state, current_actions, tensor(one_hot_option)], dim=-1))
            q2_intra_targ = self.model_target.intra_q_function_2(
                torch.cat([state, current_actions, tensor(one_hot_option)], dim=-1))
            backup_inter = torch.min(q1_intra_targ, q2_intra_targ) - self.args.alpha * logp

        # Inter-Q Function Loss
        loss_inter_q1 = ((q1_inter - backup_inter) ** 2).mean()
        loss_inter_q2 = ((q2_inter - backup_inter) ** 2).mean()
        loss_inter_q = loss_inter_q1 + loss_inter_q2
        return loss_inter_q

    def compute_loss_intra(self, state, action, option, one_hot_option, option_indices, next_state, reward, done):
        q1_intra = self.model.intra_q_function_1(torch.cat([state, action, tensor(one_hot_option)], dim=-1))
        q2_intra = self.model.intra_q_function_2(torch.cat([state, action, tensor(one_hot_option)], dim=-1))

        beta_prob, logp, current_actions = [], [], []
        for i in range(self.args.batch_size):
            option_element = option[i].to(dtype=torch.long)
            next_state_element = next_state[i]
            state_element = state[i]
            beta_prob_element, termination = self.predict_option_termination(tensor(next_state_element), option_element)
            beta_prob.append(beta_prob_element)
            current_action_element, logp_element = self.model.intra_option_policies[option_element](state_element)
            logp.append(logp_element)
            current_actions.append(current_action_element)

        beta_prob = torch.stack(beta_prob)
        logp = torch.stack(logp)
        current_actions = torch.stack(current_actions)

        with torch.no_grad():
            q1_inter_targ = self.model_target.inter_q_function_1(next_state)
            q2_inter_targ = self.model_target.inter_q_function_2(next_state)
            q_inter_targ = torch.min(q1_inter_targ, q2_inter_targ)
            q_inter_targ_current_option = torch.gather(q_inter_targ, 1, option_indices)
            next_option = torch.LongTensor(self.get_option(next_state, self.get_epsilon(), gradient=True))
            next_option = next_option.unsqueeze(-1)

            q_inter_targ_next_option = torch.gather(q_inter_targ, 1, next_option)

            backup_intra = reward + self.args.gamma * (1. - done) * (((1. - beta_prob) * q_inter_targ_current_option) +
                                                                     (beta_prob * q_inter_targ_next_option))
        # Intra-Q Function Loss
        loss_intra_q1 = ((q1_intra - backup_intra) ** 2).mean()
        loss_intra_q2 = ((q2_intra - backup_intra) ** 2).mean()
        loss_intra_q = loss_intra_q1 + loss_intra_q2

        # Intra-Option Policy Loss
        # the replay buffer, so we should not another sampling, which can be
        # different from the action sampled from the buffer
        q_pi = torch.min(q1_intra, q2_intra).detach()
        loss_intra_pi = (self.args.alpha * logp - q_pi).mean()

        return loss_intra_q, loss_intra_pi, current_actions, logp, beta_prob

    def update_loss_soc(self, data):
        state, option, action, reward, next_state, done = \
            data['state'], data['option'], data['action'], data['reward'], data['next_state'], data['done']

        one_hot_option = convert_onehot(option, self.args.option_num)
        option_indices = torch.LongTensor(option.numpy().flatten().astype(int))
        option_indices = option_indices.unsqueeze(-1)

        # Updating Intra-Q Functions
        self.intra_q_function_optim.zero_grad()
        loss_intra_q, loss_intra_pi, current_actions, logp, beta_prob = self.compute_loss_intra(
            state, action, option, one_hot_option, option_indices, next_state, reward, done)
        loss_intra_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params_intra, self.args.max_grad_clip)
        self.intra_q_function_optim.step()
        self.tb_writer.log_data("intra_q_function_loss", self.iteration, loss_intra_q.item())

        # Updating Inter-Q Functions
        self.inter_q_function_optim.zero_grad()
        loss_inter_q = self.compute_loss_inter(state, option_indices, one_hot_option, current_actions, logp)
        loss_inter_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params_inter, self.args.max_grad_clip)
        self.inter_q_function_optim.step()
        self.tb_writer.log_data("inter_q_function_loss", self.iteration, loss_inter_q.item())

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params_intra:
            p.requires_grad = False

        for p in self.q_params_inter:
            p.requires_grad = False

        # Updating Intra-Policy
        self.intra_policy_optim.zero_grad()
        loss_intra_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params_intra, self.args.max_grad_clip)
        self.intra_policy_optim.step()
        self.tb_writer.log_data("intra_q_policy_loss", self.iteration, loss_intra_pi.item())

        # Updating Beta-Policy
        self.beta_optim.zero_grad()
        loss_beta = self.compute_loss_beta(next_state, option_indices, beta_prob)
        loss_beta.backward()
        torch.nn.utils.clip_grad_norm_(self.beta_params, self.args.max_grad_clip)
        self.beta_optim.step()
        self.tb_writer.log_data("beta_policy_loss", self.iteration, loss_beta.item())

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params_intra:
            p.requires_grad = True
        for p in self.q_params_inter:
            p.requires_grad = True

        # Finally, update target inter-q networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)
