import random
import itertools
import torch.nn as nn
from copy import deepcopy
from agent_utils.models import *  # TODO No star
from torch.optim import Adam


class SoftOptionCritic(nn.Module):
    def __init__(self, observation_space, action_space, args):
        super(SoftOptionCritic, self).__init__()

        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.option_num = args.option_num

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
        self.inter_option_policy = InterOptionPolicy(self.obs_dim, self.option_num, args.hidden_dim)
        for option_index in range(self.option_num):
            self.beta_list.append(BetaPolicy(self.obs_dim, args.hidden_dim))
            self.intra_option_policies.append(
                IntraOptionPolicy(self.obs_dim, self.option_num, self.action_dim, args.hidden_dim))

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

        self.inter_policy_params = self.inter_option_policy.parameters()
        self.intra_policy_params = itertools.chain(*[policy.parameters() for policy in self.intra_option_policies])
        self.beta_params = itertools.chain(*[policy.parameters() for policy in self.beta_list])

        self.inter_q_function_optim = Adam(self.q_params_inter, lr=args.lr)
        self.intra_q_function_optim = Adam(self.q_params_intra, lr=args.lr)

        self.inter_policy_optim = Adam(self.inter_policy_params, lr=args.lr)
        self.intra_policy_optim = Adam(self.intra_policy_params, lr=args.lr)

        self.beta_optim = Adam(self.beta_params, lr=args.lr)

    def get_option(self, q, eta):
        if random.random() > eta:
            return np.argmax(np.max(q.data.numpy()))
        else:
            return random.randint(0, self.option_num - 1)

    def esoft(self, q, eta):
        if random.random() > eta:
            return np.argmax(np.max(q.data.numpy()))
        else:
            return random.randint(0, self.option_num - 1)
