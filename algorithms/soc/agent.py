import random
import torch
import gym

import itertools
import numpy as np
import torch.nn as nn

from math import exp
from copy import deepcopy
from torch.optim import Adam
from torch.distributions import Bernoulli
from misc.torch_utils import tensor
from algorithms.soc.model import SOCModel
from algorithms.soc.cnn_categorical_model import SOCModelCategorical

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
torch.autograd.set_detect_anomaly(True)

class SoftOptionCritic(nn.Module):
    def __init__(self, observation_space, action_space, args, tb_writer, log):
        super(SoftOptionCritic, self).__init__()

        self.obs_dim = observation_space.shape[0]
        self.action_space = action_space
        self.option_num = args.option_num
        self.args = args
        self.tb_writer = tb_writer
        self.log = log
        self.nonstationarity_index = 0
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
            self.model = SOCModelCategorical(self.obs_dim, self.action_dim, args.hidden_size, self.option_num)
        else:
            self.action_dim = action_space.shape[0]
            self.model = SOCModel(self.obs_dim, self.action_dim, self.args.hidden_size, self.option_num,
                                  action_space.high[0])

        self.model_target = deepcopy(self.model)

        # Parameter Definitions
        self.q_params_inter = itertools.chain(
            self.model.inter_q_function_1.parameters(),
            self.model.inter_q_function_2.parameters())

        self.q_params_intra = itertools.chain(
            self.model.intra_q_function_1.parameters(),
            self.model.intra_q_function_2.parameters())

        self.intra_policy_params = itertools.chain(self.model.intra_option_policy.parameters())
        self.beta_params = itertools.chain(self.model.beta_policy.parameters())

        self.inter_q_function_optim = Adam(self.q_params_inter, lr=args.lr)
        self.intra_q_function_optim = Adam(self.q_params_intra, lr=args.lr)
        self.intra_policy_optim = Adam(self.intra_policy_params, lr=args.lr)
        self.beta_optim = Adam(self.beta_params, lr=args.lr)

        self.iteration = 0
        self.test_iteration = 0
        self.episodes = 0
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.model_target.parameters():
            p.requires_grad = False

    def get_epsilon(self, decay=False, eval=False):
        if decay:
            eps = self.args.eps_min + (self.args.eps_start - self.args.eps_min) * exp(
                -self.iteration / self.args.eps_decay)
            self.tb_writer.log_data("epsilon", self.iteration, eps)
        elif eval:
            eps = 0.0
        else:
            eps = self.args.eps_min
        return eps

    def get_option(self, state, eta, gradient=False):  # soft-epsilon strategy
        q_function = torch.min(self.model.inter_q_function_1(tensor(state), gradient=gradient),
                               self.model.inter_q_function_2(tensor(state), gradient=gradient))

        option = np.argmax(q_function.data.numpy(), axis=-1)
        if gradient:
            return option
        else:
            if random.random() < eta:
                # option = torch.tensor([random.randint(0, self.option_num - 1)])
                option = np.array([random.randint(0, self.option_num - 1)])
            self.tb_writer.log_data("option", self.iteration, option)
            return option

    def predict_option_termination(self, state, option_indices, gradient=False):
        termination = self.model.beta_policy(state, gradient=gradient)
        if gradient:
            termination = torch.gather(termination, 1, option_indices.squeeze(-1).long()).squeeze(-1)
        else:
            termination = termination[option_indices].detach()
        option_termination = Bernoulli(termination).sample()
        return termination, option_termination == 1

    def get_action(self, option_indices, state, deterministic=False):
        action, logp = self.model.intra_option_policy(torch.as_tensor(state, dtype=torch.float32), gradient=False,
                                                      deterministic=deterministic)
        option_indices = torch.LongTensor(option_indices).unsqueeze(-1)
        action = action[option_indices:, ]
        return action.detach().numpy()[0], logp

    def compute_loss_beta(self, next_state, option_indices, beta_prob, done):
        with torch.no_grad():
            # Computing Beta Policy Loss
            q1_pi = self.model.inter_q_function_1(next_state, gradient=True)
            q2_pi = self.model.inter_q_function_2(next_state, gradient=True)
            q_pi = torch.min(q1_pi, q2_pi)

            q_pi_current_option = torch.gather(q_pi, 1, option_indices.squeeze(-1)).squeeze(-1)
            q_pi_next_option = np.max(q_pi.data.numpy())
            advantage = q_pi_current_option - q_pi_next_option

        # Beta Policy Loss
        loss_beta = ((beta_prob * advantage) * torch.logical_not(done)).mean()
        return loss_beta

    def compute_loss_inter(self, state, option_indices, one_hot_option, current_actions, logp):
        q1_inter_all = self.model.inter_q_function_1(state, gradient=True)
        q2_inter_all = self.model.inter_q_function_2(state, gradient=True)

        q1_inter = torch.gather(q1_inter_all, 1, option_indices.squeeze(-1)).squeeze(-1)
        q2_inter = torch.gather(q2_inter_all, 1, option_indices.squeeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Computing Inter-Q Function Loss
            q1_intra_targ = self.model_target.intra_q_function_1(state, tensor(one_hot_option), tensor(current_actions.unsqueeze(-1)))
            q2_intra_targ = self.model_target.intra_q_function_2(state, tensor(one_hot_option), tensor(current_actions.unsqueeze(-1)))
            backup_inter = torch.min(q1_intra_targ, q2_intra_targ) - self.args.alpha * logp

        # Inter-Q Function Loss
        loss_inter_q1 = ((q1_inter - backup_inter) ** 2).mean()
        loss_inter_q2 = ((q2_inter - backup_inter) ** 2).mean()
        loss_inter_q = loss_inter_q1 + loss_inter_q2
        return loss_inter_q

    def compute_loss_intra(self, state, action, one_hot_option, option_indices, next_state, reward, done):
        q1_intra = self.model.intra_q_function_1(state, tensor(one_hot_option), tensor(action))
        q2_intra = self.model.intra_q_function_2(state, tensor(one_hot_option), tensor(action))

        beta_prob, _ = self.predict_option_termination(tensor(next_state), option_indices, gradient=True)
        current_actions, logp = self.model.intra_option_policy(torch.as_tensor(state, dtype=torch.float32),
                                                               gradient=True)

        # current actions --> [options, batch size, actions]
        # option indices --> [batch_size, 1, 1]

        current_actions = torch.gather(current_actions.T, 1, option_indices.squeeze(-1).long()).squeeze(-1)
        logp = torch.gather(logp.reshape(self.args.batch_size, self.option_num, 1), 1, option_indices).squeeze(-1)

        with torch.no_grad():
            q1_inter_targ = self.model_target.inter_q_function_1(next_state, gradient=True)
            q2_inter_targ = self.model_target.inter_q_function_2(next_state, gradient=True)
            q_inter_targ = torch.min(q1_inter_targ, q2_inter_targ)
            q_inter_targ_current_option = torch.gather(q_inter_targ, 1, option_indices.squeeze(-1)).squeeze(-1)

            next_option = torch.tensor(self.get_option(next_state, self.get_epsilon(), gradient=True))

            q_inter_targ_next_option = torch.gather(q_inter_targ, 1, next_option.unsqueeze(-1)).squeeze(-1)

            backup_intra = reward + self.args.gamma * torch.logical_not(done) * (((1. - beta_prob) * q_inter_targ_current_option) +
                                                                     (beta_prob * q_inter_targ_next_option)).sum(dim=-1)
        # Intra-Q Function Loss
        loss_intra_q1 = ((q1_intra - backup_intra.detach()) ** 2).mean()
        loss_intra_q2 = ((q2_intra - backup_intra.detach()) ** 2).mean()
        loss_intra_q = loss_intra_q1 + loss_intra_q2

        # Intra-Option Policy Loss
        # the replay buffer, so we should not another sampling, which can be
        # different from the action sampled from the buffer

        q1_intra_current_action = self.model.intra_q_function_1(state, tensor(one_hot_option), tensor(current_actions.unsqueeze(-1)))
        q2_intra_current_action = self.model.intra_q_function_2(state, tensor(one_hot_option), tensor(current_actions.unsqueeze(-1)))
        q_pi = torch.min(q1_intra_current_action, q2_intra_current_action)

        loss_intra_pi = (self.args.alpha * logp.squeeze(-1) - q_pi).mean()

        return loss_intra_q, loss_intra_pi, current_actions, logp, beta_prob

    def update_loss_soc(self, data):
        state, option, action, reward, next_state, done = data

        state = torch.tensor(state, device=device)
        next_state = torch.tensor(next_state, device=device)
        option = torch.as_tensor(option, dtype=torch.float32)
        done = torch.tensor(done, device=device)
        reward = torch.tensor(reward, device=device)
        action = torch.tensor(action, device=device).unsqueeze(-1)
        one_hot_option = torch.nn.functional.one_hot(option.squeeze(-1).long())

        option_indices = torch.tensor(option, dtype=torch.int64).unsqueeze(-1).long()
        # Updating Intra-Q Functions
        self.intra_q_function_optim.zero_grad()
        loss_intra_q, loss_intra_pi, current_actions, logp, beta_prob = self.compute_loss_intra(
            state, action, one_hot_option, option_indices, next_state, reward, done)
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

        # Updating Intra-Policy
        self.intra_policy_optim.zero_grad()
        loss_intra_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.intra_policy_params, self.args.max_grad_clip)
        self.intra_policy_optim.step()
        self.tb_writer.log_data("intra_q_policy_loss", self.iteration, loss_intra_pi.item())

        # Updating Beta-Policy
        self.beta_optim.zero_grad()
        loss_beta = self.compute_loss_beta(next_state, option_indices, beta_prob, done)
        loss_beta.backward()
        torch.nn.utils.clip_grad_norm_(self.beta_params, self.args.max_grad_clip)
        self.beta_optim.step()
        self.tb_writer.log_data("beta_policy_loss", self.iteration, loss_beta.item())
        self.iteration += 1

        with torch.no_grad():
            for p, p_targ in zip(self.model.inter_q_function_1.parameters(),
                                 self.model_target.inter_q_function_1.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

            for p, p_targ in zip(self.model.inter_q_function_2.parameters(),
                                 self.model_target.inter_q_function_2.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

            for p, p_targ in zip(self.model.intra_q_function_1.parameters(),
                                 self.model_target.intra_q_function_1.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

            for p, p_targ in zip(self.model.intra_q_function_2.parameters(),
                                 self.model_target.intra_q_function_2.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)
