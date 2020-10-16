import torch
from torch import nn as nn
import torch.multiprocessing
import torch.optim as optim
from collections import defaultdict
import math
import numpy as np


class SharedAdam(optim.Optimizer):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-3,
                 weight_decay=0,
                 amsgrad=False):
        defaults = defaultdict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['max_exp_avg_sq'] = p.data.new().resize_as_(
                    p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['max_exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till
                    # now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step'].item()
                bias_correction2 = 1 - beta2**state['step'].item()
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_options=4):
        super(ActorCritic, self).__init__()
        self.module_list = nn.ModuleList()

        self.layer1 = nn.Linear(num_inputs, 64)
        self.module_list += [self.layer1]
        self.nonlin1 = nn.Tanh()
        self.layer2 = nn.Linear(64, 64)
        self.module_list += [self.layer2]
        self.nonlin2 = nn.Tanh()
        self.layer3 = nn.Linear(64, 64)
        self.module_list += [self.layer3]
        self.nonlin3 = nn.Tanh()

        self.critic_linear = nn.Linear(64, num_options)
        self.module_list += [self.critic_linear]

        self.apply(weights_init)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.policylayer = {}
        self.termlayer = {}
        for i in range(0, num_options):
            self.policylayer[i] = nn.Linear(64, num_outputs)
            self.module_list += [self.policylayer[i]]
            self.termlayer[i] = nn.Linear(64, 1)
            self.module_list += [self.termlayer[i]]
            self.policylayer[i].weight.data = norm_col_init(self.policylayer[i].weight.data, 0.01)
            self.policylayer[i].bias.data.fill_(0)
            self.termlayer[i].weight.data = norm_col_init(self.termlayer[i].weight.data, 0.01)
            self.termlayer[i].bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = self.nonlin1(self.layer1(inputs))
        x = self.nonlin2(self.layer2(x))
        x = self.nonlin3(self.layer3(x))

        value = self.critic_linear(x)
        return value, x

    def getTermination(self, hidden, o):
        term = torch.sigmoid(self.termlayer[o](hidden))
        return term

    def getAction(self, hidden, o):
        action = self.policylayer[o](hidden)
        return action
