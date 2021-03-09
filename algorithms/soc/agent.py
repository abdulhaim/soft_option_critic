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
import torch.nn.functional as F
from misc.torch_utils import convert_onehot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class SoftOptionCritic(nn.Module):
    def __init__(self, observation_space, action_space, args, tb_writer, log):
        super(SoftOptionCritic, self).__init__()
        self.observation_space = observation_space
        if isinstance(observation_space, gym.spaces.Discrete):
            self.obs_dim = 1
        else:
            self.obs_dim = observation_space.shape[0]
        self.option_num = args.option_num
        self.args = args
        self.tb_writer = tb_writer
        self.log = log
        self.nonstationarity_index = 0
        self.action_space = action_space

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = 1
            if len(observation_space.shape) == 3:
                from algorithms.soc.cnn_categorical_model import SOCModelCategorical
                self.model = SOCModelCategorical(self.obs_dim, self.action_space.n, args.hidden_size, self.option_num)
            else:
                from algorithms.soc.categorical_model import SOCModelCategorical
                self.model = SOCModelCategorical(self.obs_dim, self.action_space, args.hidden_size, self.option_num, args.num_experts, args.moe_hidden_size, args.top_k, self.args.batch_size)

        else:
            self.action_dim = action_space.shape[0]
            from algorithms.soc.model import SOCModel
            self.model = SOCModel(self.obs_dim, self.action_dim, self.args.hidden_size, self.option_num,
                                  action_space.high[0])

        self.model = self.model.to(device)
        self.model_target = deepcopy(self.model).to(device)

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
        q_function = torch.min(self.model.inter_q_function_1(tensor(state).to(device), gradient=gradient),
                               self.model.inter_q_function_2(tensor(state).to(device), gradient=gradient))

        if self.args.option_num != 1:
            option = np.argmax(q_function.data.cpu().numpy(), axis=-1)
        else:
            if gradient:
                option = np.full(shape=self.args.batch_size, fill_value=0, dtype=np.int)
            else:
                option = 0
        if gradient:
            return option
        else:
            if random.random() < eta:
                option = random.randint(0, self.option_num - 1)
            return option

    def predict_option_termination(self, state, option_indices, gradient=False):
        termination = self.model.beta_policy(state, gradient=gradient)
        if gradient:
            termination = torch.gather(termination, 1, option_indices).squeeze(-1)
        else:
            termination = termination[option_indices].detach()
        option_termination = Bernoulli(termination).sample()
        return termination, option_termination == 1

    def get_action(self, option_indices, state, deterministic=False, with_logprob=False):
        action, logp = self.model.intra_option_policy(torch.as_tensor(state, dtype=torch.float32), option_indices,
                                                      deterministic=deterministic, with_logprob=with_logprob)

        if isinstance(self.action_space, gym.spaces.Discrete):
            return action.numpy()[0], logp
        else:
            option_indices = torch.tensor(option_indices, dtype=torch.long).unsqueeze(-1)
            action = action.squeeze(-1)
            action = torch.gather(action, 0, option_indices).squeeze(-1)
            return np.array([action.detach().cpu()]), logp

    def compute_loss_beta(self, next_state, option_indices, beta_prob, done):
        # Computing Inter Q Values for Advantage
        with torch.no_grad():
            q1_pi = self.model.inter_q_function_1(next_state, gradient=True)
            q2_pi = self.model.inter_q_function_2(next_state, gradient=True)
            q_pi = torch.min(q1_pi, q2_pi)

            if self.args.option_num == 1:
                q_pi = q_pi.unsqueeze(-1)

            assert q_pi.shape == (self.args.batch_size, self.option_num)

            q_pi_current_option = torch.gather(q_pi, 1, option_indices)
            q_pi_next_option = torch.max(q_pi, dim=1)[0].unsqueeze(-1)

            assert q_pi_current_option.shape == (self.args.batch_size, 1)
            assert q_pi_next_option.shape == (self.args.batch_size, 1)

            advantage = q_pi_current_option - q_pi_next_option
            assert advantage.shape == (self.args.batch_size, 1)
            assert torch.logical_not(done).shape == (self.args.batch_size, 1)

        # Beta Policy Loss
        loss_beta = ((beta_prob * advantage) * torch.logical_not(done)).mean()
        return loss_beta

    def compute_loss_inter(self, state, option_indices, one_hot_option, current_actions, logp):
        # Computer Inter-Q Values
        q1_inter_all = self.model.inter_q_function_1(state, gradient=True)
        q2_inter_all = self.model.inter_q_function_2(state, gradient=True)

        if self.args.option_num == 1:
            q1_inter_all = q1_inter_all.unsqueeze(-1)
            q2_inter_all = q2_inter_all.unsqueeze(-1)

        assert q1_inter_all.shape == (self.args.batch_size, self.option_num)

        q1_inter = torch.gather(q1_inter_all, 1, option_indices)
        q2_inter = torch.gather(q2_inter_all, 1, option_indices)

        assert q1_inter.shape == (self.args.batch_size, 1)

        # Intra Q Values for Target
        with torch.no_grad():
            q1_intra_targ = self.model_target.intra_q_function_1(state, one_hot_option, current_actions)
            q2_intra_targ = self.model_target.intra_q_function_2(state, one_hot_option, current_actions)
            q_intra_targ = torch.min(q1_intra_targ, q2_intra_targ).unsqueeze(-1)

            # Inter-Q Back Up Values
            if isinstance(self.action_space, gym.spaces.Discrete):
                q_intra_targ = q_intra_targ.squeeze(-1)
                assert q_intra_targ.shape == (self.args.batch_size, self.action_space.n)
                backup_inter = (current_actions * (q_intra_targ - (self.args.alpha * logp))).sum(dim=-1)
                backup_inter = torch.unsqueeze(backup_inter, -1)

            else:
                assert q_intra_targ.shape == (self.args.batch_size, 1)
                backup_inter = q_intra_targ - (self.args.alpha * logp)

            assert backup_inter.shape == (self.args.batch_size, 1)

        # Inter-Q Function Loss
        loss_inter_q1 = F.mse_loss(q1_inter, backup_inter)
        loss_inter_q2 = F.mse_loss(q2_inter, backup_inter)

        loss_inter_q = loss_inter_q1 + loss_inter_q2

        logp = torch.unsqueeze(logp, -1)
        entropy_debug = (-current_actions.unsqueeze(-1) * logp).sum(dim=-1).mean()
        self.tb_writer.log_data("loss/entropy", self.iteration, entropy_debug)

        return loss_inter_q

    def compute_loss_intra_q(self, state, action, one_hot_option, option_indices, next_state, reward, done):
        # Computing Intra Q Values
        if isinstance(self.action_space, gym.spaces.Discrete):
            q1_intra = self.model.intra_q_function_1(state, one_hot_option)
            q2_intra = self.model.intra_q_function_2(state, one_hot_option)
            assert q1_intra.shape == (self.args.batch_size, self.action_space.n)

        else:
            q1_intra = self.model.intra_q_function_1(state, one_hot_option, action).unsqueeze(-1)
            q2_intra = self.model.intra_q_function_2(state, one_hot_option, action).unsqueeze(-1)
            assert q1_intra.shape == (self.args.batch_size, 1)

        # Beta Prob for Target
        beta_prob, _ = self.predict_option_termination(next_state, option_indices, gradient=True)
        beta_prob = beta_prob.unsqueeze(-1)
        assert beta_prob.shape == (self.args.batch_size, 1)

        # Inter Q Values for Target
        with torch.no_grad():
            q1_inter_targ = self.model_target.inter_q_function_1(next_state, gradient=True)
            q2_inter_targ = self.model_target.inter_q_function_2(next_state, gradient=True)
            q_inter_targ = torch.min(q1_inter_targ, q2_inter_targ)

            if self.args.option_num == 1:
                q_inter_targ = q_inter_targ.unsqueeze(-1)

            assert q_inter_targ.shape == (self.args.batch_size, self.option_num)

            q_inter_targ_current_option = torch.gather(q_inter_targ, 1, option_indices)
            next_option = torch.tensor(self.get_option(next_state, self.get_epsilon(), gradient=True))
            q_inter_targ_next_option = torch.gather(q_inter_targ, 1, next_option.unsqueeze(-1))

            assert q_inter_targ_current_option.shape == (self.args.batch_size, 1)
            assert q_inter_targ_next_option.shape == (self.args.batch_size, 1)

            # Intra-Q Back Up Values
            backup_intra = (reward + self.args.gamma * torch.logical_not(done) * (
                    ((1. - beta_prob) * q_inter_targ_current_option) +
                    (beta_prob * q_inter_targ_next_option)))

            assert backup_intra.shape == (self.args.batch_size, 1)

        # Computing Intra Q Loss
        if isinstance(self.action_space, gym.spaces.Discrete):
            loss_intra_q1 = F.mse_loss(torch.gather(q1_intra, 1, action.long()), backup_intra.detach())
            loss_intra_q2 = F.mse_loss(torch.gather(q2_intra, 1, action.long()), backup_intra.detach())
        else:
            loss_intra_q1 = F.mse_loss(q1_intra, backup_intra.detach())
            loss_intra_q2 = F.mse_loss(q1_intra, backup_intra.detach())

        loss_intra_q = loss_intra_q1 + loss_intra_q2

        return loss_intra_q, beta_prob

    def compute_loss_intra_policy(self, state, option_indices, one_hot_option):
        # Sampling Actions from Policy
        current_actions, logp = self.model.intra_option_policy(state, option_indices, deterministic=False)

        if isinstance(self.action_space, gym.spaces.Discrete):
            assert current_actions.shape == (self.args.batch_size, self.action_space.n)
            assert logp.shape == (self.args.batch_size, self.action_space.n)
        else:
            assert current_actions.shape == (self.args.batch_size, self.action_dim)
            assert logp.shape == (self.args.batch_size, self.action_dim)

        # Computing Intra Q Values with Sampled Actions
        q1_intra_current_action = self.model.intra_q_function_1(state, one_hot_option, current_actions)
        q2_intra_current_action = self.model.intra_q_function_2(state, one_hot_option, current_actions)
        q_pi = torch.min(q1_intra_current_action, q2_intra_current_action).unsqueeze(-1)

        # Intra-Policy Loss
        if isinstance(self.action_space, gym.spaces.Discrete):
            q_pi = q_pi.squeeze(-1)
            assert q_pi.shape == (self.args.batch_size, self.action_space.n)
            loss_intra_pi = (current_actions * (self.args.alpha * logp - q_pi)).sum(dim=1).mean()
        else:
            assert q_pi.shape == (self.args.batch_size, 1)
            loss_intra_pi = (self.args.alpha * logp - q_pi).mean()

        return loss_intra_pi, current_actions, logp

    def update_loss_soc(self, data):
        state, option, action, reward, next_state, done = data['state'], data['option'], data['action'], data['reward'], data['next_state'], data['done']

        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)

        one_hot_option = torch.tensor(convert_onehot(option, self.args.option_num), dtype=torch.float32)
        option_indices = torch.LongTensor(option.cpu().numpy().flatten().astype(int)).unsqueeze(-1).to(device)

        # Updating Intra-Q
        loss_intra_q, beta_prob = self.compute_loss_intra_q(state, action, one_hot_option, option_indices, next_state, reward, done)
        loss_intra_pi, current_actions, logp = self.compute_loss_intra_policy(state, option_indices, one_hot_option)
        self.intra_q_function_optim.zero_grad()
        loss_intra_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params_intra, self.args.max_grad_clip)
        self.intra_q_function_optim.step()
        self.tb_writer.log_data("intra_q_function_loss", self.iteration, loss_intra_q.item())

        # Updating Intra-Policy
        self.intra_policy_optim.zero_grad()
        loss_intra_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.intra_policy_params, self.args.max_grad_clip)
        self.intra_policy_optim.step()
        self.tb_writer.log_data("intra_q_policy_loss", self.iteration, loss_intra_pi.item())

        # Updating Inter-Q
        loss_inter_q = self.compute_loss_inter(state, option_indices, one_hot_option, current_actions, logp)
        self.inter_q_function_optim.zero_grad()
        loss_inter_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params_inter, self.args.max_grad_clip)
        self.inter_q_function_optim.step()
        self.tb_writer.log_data("inter_q_function_loss", self.iteration, loss_inter_q.item())

        # Updating Beta-Policy
        loss_beta = self.compute_loss_beta(next_state, option_indices, beta_prob, done)
        self.beta_optim.zero_grad()
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

