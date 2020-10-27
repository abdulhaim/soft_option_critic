import itertools
import torch.nn as nn
from copy import deepcopy
from agent_utils.models import *
from torch.optim import Adam


class SoftActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, args):
        super(SoftActorCritic, self).__init__()

        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.args = args

        # Q Function Definitions
        self.q_function_1 = QFunction(self.obs_dim, self.action_dim, args.hidden_dim)
        self.q_function_2 = QFunction(self.obs_dim, self.action_dim, args.hidden_dim)
        self.q_function_1_targ = deepcopy(self.q_function_1)
        self.q_function_2_targ = deepcopy(self.q_function_2)

        # Policy Definition
        self.policy = Policy(self.obs_dim, self.action_dim, args.hidden_dim)

        # Parameter Definitions
        self.q_params = itertools.chain(self.q_function_1.parameters(), self.q_function_1.parameters())
        self.q_params_target = itertools.chain(self.q_function_1_targ.parameters(), self.q_function_1_targ.parameters())
        self.policy_params = self.policy.parameters()

        self.pi_optimizer = Adam(self.policy_params, lr=args.lr)
        self.q_optimizer = Adam(self.q_params, lr=args.lr)
