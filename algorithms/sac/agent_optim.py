import itertools
import torch
import random
import gym
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from algorithms.sac.model import SACModel
from algorithms.sac.cnn_categorical_model import SACModelCategorical
from collections import OrderedDict
from torch.optim import Adam

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

class SoftActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, args, tb_writer, log):
        super(SoftActorCritic, self).__init__()
        self.action_space = action_space
        self.obs_dim = observation_space
        self.args = args
        self.tb_writer = tb_writer
        self.log = log

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
            self.model = SACModelCategorical(observation_space, action_space, args.hidden_size)
        else:
            self.action_dim = action_space.shape[0]
            self.model = SACModel(observation_space, action_space, args.hidden_size)
            self.model.to(device)

        self.model_target = deepcopy(self.model)
        self.model_target.to(device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.model_target.parameters():
            p.requires_grad = False

        if args.mer:
            self.lr = args.mer_lr
        else:
            self.lr = args.lr
        # Parameter Definitions
        # self.q_params = itertools.chain(self.model.q_function_1.parameters(), self.model.q_function_2.parameters())
        # if args.mer:
        #     self.pi_optimizer = Adam(self.model.policy.parameters(), lr=args.mer_lr)
        #     self.q_optimizer = Adam(self.q_params, lr=args.mer_lr)
        # else:
        #     self.pi_optimizer = Adam(self.model.policy.parameters(), lr=args.lr)
        #     self.q_optimizer = Adam(self.q_params, lr=args.lr)

        self.test_iteration = 0
        self.iteration = 0
        self.episodes = 0
        self.nonstationarity_index = 0

    def get_current_sample(self):
        return self.current_sample

    def compute_loss_q(self, data):
        o, a, r, o2, d = data
        o = torch.tensor(o, device=device)
        o2 = torch.tensor(o2, device=device)
        d = torch.tensor(d, device=device)
        r = torch.tensor(r, device=device)
        a = torch.tensor(a, device=device)

        # o, a, r, o2, d = data['state'], data['action'], data['reward'], data['next_state'], data['done']
        if isinstance(self.action_space, gym.spaces.Discrete):
            q1 = self.model.q_function_1(o)
            q2 = self.model.q_function_2(o)
        else:
            q1 = self.model.q_function_1(o, a)
            q2 = self.model.q_function_2(o, a)

        if isinstance(self.action_space,
                      gym.spaces.Discrete) and not self.args.mer:  # TODO: add function to check dimensions for different cases
            assert q1.shape == (self.args.batch_size, self.action_dim)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.policy(o2, with_logprob=True)

            # Target Q-values
            if isinstance(self.action_space, gym.spaces.Discrete):
                q1_pi_targ = self.model_target.q_function_1(o2)
                q2_pi_targ = self.model_target.q_function_2(o2)
            else:
                q1_pi_targ = self.model_target.q_function_1(o2, a2)
                q2_pi_targ = self.model_target.q_function_2(o2, a2)

            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            if isinstance(self.action_space, gym.spaces.Discrete):
                backup = r + self.args.gamma * torch.logical_not(d) * (
                        a2 * (q_pi_targ - self.args.alpha * logp_a2)).sum(dim=-1)
                backup = torch.unsqueeze(backup, -1)

            else:
                backup = r + self.args.gamma * (1 - d) * (q_pi_targ - self.args.alpha * logp_a2)

        # MSE loss against Bellman backup
        if isinstance(self.action_space,
                      gym.spaces.Discrete) and not self.args.mer:  # TODO: add function to check dimensions for different cases
            assert backup.shape == (self.args.batch_size, 1)

        if isinstance(self.action_space, gym.spaces.Discrete):
            loss_q1 = F.mse_loss(torch.gather(q1, 1, a.unsqueeze(-1).long()), backup)
            loss_q2 = F.mse_loss(torch.gather(q2, 1, a.unsqueeze(-1).long()), backup)
        else:
            loss_q1 = F.mse_loss(q1, backup)
            loss_q2 = F.mse_loss(q2, backup)

        loss_q = loss_q1 + loss_q2

        logp_a2 = torch.unsqueeze(logp_a2, -1)

        entropy_debug = (-a2.unsqueeze(-1) * logp_a2).sum(dim=-1).mean()
        self.tb_writer.log_data("loss/entropy", self.iteration, entropy_debug)

        return loss_q

    def compute_loss_pi(self, data):
        o = data[0]
        o = torch.tensor(o, device=device)
        pi, logp_pi = self.model.policy(o, with_logprob=True)

        if isinstance(self.action_space, gym.spaces.Discrete):
            q1_pi = self.model.q_function_1(o).detach()
            q2_pi = self.model.q_function_2(o).detach()
        else:
            q1_pi = self.model.q_function_1(o, pi)
            q2_pi = self.model.q_function_2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        if isinstance(self.action_space, gym.spaces.Discrete):
            loss_pi = (pi * (self.args.alpha * logp_pi - q_pi)).sum(dim=1).mean()
        else:
            loss_pi = (self.args.alpha * logp_pi - q_pi).mean()

        return loss_pi

    def update_loss_sac(self, data):
        # First run one gradient descent step for Q1 and Q2
        loss_q = self.compute_loss_q(data)
        q1_grad = torch.autograd.grad(loss_q, self.model.q_function_1.parameters(), create_graph=False)
        q2_grad = torch.autograd.grad(loss_q, self.model.q_function_2.parameters(), create_graph=False)

        phi_q_1 = OrderedDict()
        phi_q_2 = OrderedDict()

        for (name, param), grad in zip(self.q_params, q_grad):
            phi_q_1[name] = param - self.lr * grad

        self.model.q_function_1.load_state_dict(phi_q)
        self.model.q_function_2.load_state_dict(phi_q)

        self.tb_writer.log_data("loss/q_function_loss", self.iteration, loss_q.item())

        # # Freeze Q-networks so you don't waste computational effort
        # # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi
        loss_pi = self.compute_loss_pi(data)
        pi_grad = torch.autograd.grad(loss_pi, self.model.policy.parameters(), create_graph=False)
        phi_pi = OrderedDict()
        for (name, param), grad in zip(self.model.policy.parameters(), pi_grad):
            phi_pi[name] = param - self.lr * grad

        self.model.policy.load_state_dict(phi_pi)
        self.tb_writer.log_data("loss/policy_loss", self.iteration, loss_pi.item())

        # # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        self.update_target_networks()

    def update_sac_mer(self, replay_buffer):
        random_index = random.randint(1, self.args.mer_replay_batch_size)
        # get current weights
        weights_before_batch = deepcopy(self.model.state_dict())
        for j_step in range(self.args.mer_replay_batch_size):
            if j_step == random_index:
                data = self.get_current_sample()
            else:
                data = replay_buffer.sample_batch(1)

            # First run one gradient descent step for Q1 and Q2
            loss_q = self.compute_loss_q(data)
            self.q_optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.q_params, self.args.max_grad_clip)
            self.q_optimizer.step()
            self.tb_writer.log_data("loss/q_function_loss", self.iteration, loss_q.item())

            # Next run one gradient descent step for pi
            loss_pi = self.compute_loss_pi(data)
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.args.max_grad_clip)
            self.pi_optimizer.step()
            self.tb_writer.log_data("loss/policy_loss", self.iteration, loss_pi.item())

        # within batch reptile meta-update
        weights_after_batch = deepcopy(self.model.state_dict())
        self.model.load_state_dict({name: weights_before_batch[name] + (
                (weights_after_batch[name] - weights_before_batch[name]) * self.args.mer_gamma) for
                                    name in weights_before_batch})

        self.update_target_networks()

    def update_target_networks(self):
        # Reset target action-value network to real action-value network after a certain number of episodes
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

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.tensor(state, device=device)
            action, logprob = self.model.act(state, deterministic)
            return action, logprob

    def load_model(self, model_dir):
        self.model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
