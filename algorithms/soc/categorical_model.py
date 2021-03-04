import torch
import torch.multiprocessing
from torch.distributions.categorical import Categorical

from torch import nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from algorithms.soc.mlp import MLP
from algorithms.soc.moe import SparseDispatcher

class InterQFunction(torch.nn.Module):
    """
    There will only be one of these q functions
    Input: state, option
    Output: 1
    """

    def __init__(self, obs_dim, option_dim, hidden_size):
        super(InterQFunction, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, option_dim)
        )

    def forward(self, obs, gradient=True):
        q = self.q(obs)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class IntraQFunction(torch.nn.Module):
    """
    There will be two q functions: q1 and q2
    Input: state, option, action
    Output: 1
    """

    def __init__(self, obs_dim, act_dim, option_dim, hidden_size):
        super(IntraQFunction, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + option_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, state, option, gradient=True):
        q = self.q(torch.cat([state, option], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class IntraOptionPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: number of actions
    """

    def __init__(self, obs_dim, act_dim, option_dim, hidden_size, num_experts, moe_hidden_size, k):
        super(IntraOptionPolicy, self).__init__()

        self.num_experts = num_experts
        self.moe_hidden_size = moe_hidden_size
        self.k = k
        # instantiate experts

        self.experts = nn.ModuleList(
            [MLP(obs_dim, act_dim, self.moe_hidden_size) for i in range(self.num_experts)])

        self.w_gate = nn.Parameter(torch.zeros(option_dim, obs_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(option_dim, obs_dim, num_experts), requires_grad=True)

        self.last_w_layer = torch.randn((option_dim, hidden_size, act_dim))
        self.last_b_layer = torch.randn((option_dim, act_dim))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(1)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, obs, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # choosing which experts based on observation

        logits = obs @ self.w_gate
        # calculate topk + 1 experts that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        load = self._gates_to_load(gates)
        return gates, load

    def forward(self, obs, gradient=False, deterministic=False, with_logprob=True, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(obs, gradient)
        # calculate importance loss
        importance = gates.sum(1)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(obs)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        mu = torch.matmul(obs, self.last_w_layer)

        if gradient:
            b_mu = torch.unsqueeze(self.last_b_layer, axis=1)
        else:
            b_mu = self.last_b_layer

        mu = torch.add(mu, b_mu)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = torch.argmax(mu, dim=-1)
        else:
            action_probs = F.softmax(mu, dim=-1)
            action_distribution = Categorical(probs=action_probs)
            action = action_distribution.sample().cpu()

        if with_logprob:
            z = (action_probs == 0.0).float() * 1e-8
            log_action_probabilities = torch.log(action_probs + z)
            return action_probs, log_action_probabilities
        else:
            log_action_probabilities = None
            return action, log_action_probabilities


class BetaPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: 1
    """

    def __init__(self, obs_dim, option_dim, hidden_size):
        super(BetaPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, option_dim))

    def forward(self, inputs, gradient=True):
        x = self.net(inputs)
        return torch.sigmoid(x)


class SOCModelCategorical(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_size, option_dim, num_experts, moe_hidden_size, k):
        super(SOCModelCategorical, self).__init__()
        # Inter-Q Function Definitions
        self.inter_q_function_1 = InterQFunction(obs_dim, option_dim, hidden_size)
        self.inter_q_function_2 = InterQFunction(obs_dim, option_dim, hidden_size)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(obs_dim, action_space.n, option_dim, hidden_size)
        self.intra_q_function_2 = IntraQFunction(obs_dim, action_space.n, option_dim, hidden_size)

        # Policy Definitions
        self.intra_option_policy = IntraOptionPolicy(obs_dim, action_space.n, option_dim, hidden_size, num_experts, moe_hidden_size, k)
        self.beta_policy = BetaPolicy(obs_dim, option_dim, hidden_size)
