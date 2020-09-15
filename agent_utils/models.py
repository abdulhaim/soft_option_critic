from agent_utils.utils import *
import torch.multiprocessing
from agent_utils.replay_buffer import *

import torch
torch.set_num_threads(8)
print(torch.get_num_threads())
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

np.random.seed(1)
torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
use_cuda = False
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if use_cuda else "cpu")

def softmax(values, temp=0.3):
    if len(values.shape) == 1:
        values = values.reshape(-1, values.shape[0])
    actual_values = values
    values = values / temp
    probs = F.softmax(values, dim=-1)
    selected = torch.multinomial(probs, num_samples=1)
    return selected, probs

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, init_w=1e-3):
        super(ActorNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, init_w=1e-3):
        super(ActorNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x


class StochasticActorNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, init_w=1e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(StochasticActorNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        num_inputs = state_dim
        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

        self.minm = minm
        self.maxm = maxm
        self.scale = scale
        self.to(device)

    def forward(self, state, epsilon=1e-6):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def scale_action(self, action):

        action = self.minm + (action + 1.0) * 0.5 * (self.maxm - self.minm)
        action = np.clip(action, self.minm, self.maxm)

        return np.array([int(round(float(action)))])

    def get_action(self, state):
        state = state.unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()
        if (torch.isnan(action[0][0])):
            import ipdb;
            ipdb.set_trace()
        if (self.scale == True):
            return self.scale_action(action.detach())
        else:
            return action[0].detach()


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=1e-3):
        super(CriticNetwork, self).__init__()
        self.q1_linear1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.q1_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.q1_linear3 = nn.Linear(hidden_dim[1], 1)
        self.q1_linear3.weight.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.q1_linear1.weight)
        torch.nn.init.xavier_uniform_(self.q1_linear2.weight)

        self.q2_linear1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.q2_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.q2_linear3 = nn.Linear(hidden_dim[1], 1)
        torch.nn.init.xavier_uniform_(self.q2_linear1.weight)
        torch.nn.init.xavier_uniform_(self.q2_linear2.weight)
        self.q2_linear3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        input = torch.cat((state, action), -1)
        x = F.relu(self.q1_linear1(input))
        x = F.relu(self.q1_linear2(x))
        q1_value = self.q1_linear3(x)

        x = F.relu(self.q2_linear1(input))
        x = F.relu(self.q2_linear2(x))
        q2_value = self.q2_linear3(x)
        return q1_value, q2_value


class OptionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, options_num, hidden_dim, vat_noise, init_w=1e-3):
        super(OptionNetwork, self).__init__()
        self.enc_linear1 = nn.Linear(state_dim + action_dim + action_dim, hidden_dim[0])
        self.enc_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.enc_linear3 = nn.Linear(hidden_dim[1], options_num)

        torch.nn.init.xavier_uniform_(self.enc_linear1.weight)
        torch.nn.init.xavier_uniform_(self.enc_linear2.weight)
        self.enc_linear3.weight.data.uniform_(-init_w, init_w)

        self.dec_linear3 = nn.Linear(hidden_dim[0], state_dim + action_dim + action_dim)
        self.dec_linear2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.dec_linear1 = nn.Linear(options_num, hidden_dim[1])
        self.dec_linear3.weight.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.dec_linear1.weight)
        torch.nn.init.xavier_uniform_(self.dec_linear2.weight)

        self.vat_noise = vat_noise

    def forward(self, state, action, mean, log_std):
        input = torch.cat((state, mean), -1)
        input = torch.cat((input, log_std), -1)

        x = F.relu(self.enc_linear1(input))
        x = F.relu(self.enc_linear2(x))
        enc_output = self.enc_linear3(x)

        output_options = F.softmax(enc_output, -1)
        inpshape = input.shape
        epsilon = Normal(0, 1).sample(inpshape)
        x2 = input + self.vat_noise * epsilon * torch.abs(input)

        input_noise = self.enc_linear1(x2)
        hidden2_noise = self.enc_linear2(input_noise)
        out_noise = self.enc_linear3(hidden2_noise)

        output_option_noise = F.softmax(out_noise, -1)

        dec_1 = F.relu(self.dec_linear1(enc_output))
        dec_2 = F.relu(self.dec_linear2(dec_1))
        dec_output = self.dec_linear3(dec_2)
        return enc_output, output_options, output_option_noise, dec_output, input

class StochasticOptionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, options_num, hidden_dim, vat_noise, init_w=1e-3, num_latent=4):
        super(StochasticOptionNetwork, self).__init__()
        self.enc_linear1 = nn.Linear(state_dim + action_dim + action_dim, hidden_dim[0])
        self.enc_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.enc_linear3 = nn.Linear(hidden_dim[1], options_num)

        torch.nn.init.xavier_uniform_(self.enc_linear1.weight)
        torch.nn.init.xavier_uniform_(self.enc_linear2.weight)
        self.enc_linear3.weight.data.uniform_(-init_w, init_w)

        self.dec_linear3 = nn.Linear(hidden_dim[0], state_dim + action_dim + action_dim)
        self.dec_linear2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.dec_linear1 = nn.Linear(options_num, hidden_dim[1])
        self.dec_linear3.weight.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.dec_linear1.weight)
        torch.nn.init.xavier_uniform_(self.dec_linear2.weight)

        self.clusters = torch.zeros((num_latent, options_num), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.clusters)

        self.vat_noise = vat_noise

    def forward(self, state, action, mean, log_std):
        input = torch.cat((state, mean), -1)
        input = torch.cat((input, log_std), -1)

        x = F.relu(self.enc_linear1(input))
        x = F.relu(self.enc_linear2(x))
        enc_output = self.enc_linear3(x)

        output_options = F.softmax(enc_output, -1)

        inpshape = input.shape
        epsilon = Normal(0, 1).sample(inpshape)
        x2 = input + self.vat_noise * epsilon * torch.abs(input)

        input_noise = self.enc_linear1(x2)
        hidden2_noise = self.enc_linear2(input_noise)
        out_noise = self.enc_linear3(hidden2_noise)

        output_option_noise = F.softmax(out_noise, -1)

        dec_1 = F.relu(self.dec_linear1(enc_output))
        dec_2 = F.relu(self.dec_linear2(dec_1))
        dec_output = self.dec_linear3(dec_2)
        return enc_output, output_options, output_option_noise, dec_output, input

class StochasticOptionNetworkKMeans(nn.Module):
    def __init__(self, state_dim, action_dim, options_num, hidden_dim, vat_noise, init_w=1e-3, num_latent=4):
        super(StochasticOptionNetworkKMeans, self).__init__()
        self.enc_linear1 = nn.Linear(state_dim + action_dim + action_dim, hidden_dim[0])
        self.enc_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.enc_linear3 = nn.Linear(hidden_dim[1], options_num)
        self.num_clusters = options_num
        self.alpha = 1.0

        torch.nn.init.xavier_uniform_(self.enc_linear1.weight)
        torch.nn.init.xavier_uniform_(self.enc_linear2.weight)
        self.enc_linear3.weight.data.uniform_(-init_w, init_w)

        self.dec_linear3 = nn.Linear(hidden_dim[0], state_dim + action_dim + action_dim)
        self.dec_linear2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.dec_linear1 = nn.Linear(options_num, hidden_dim[1])
        self.dec_linear3.weight.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.dec_linear1.weight)
        torch.nn.init.xavier_uniform_(self.dec_linear2.weight)

        self.clusters = torch.zeros((num_latent, options_num), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.clusters)
        self.cluster_set = False

        self.vat_noise = vat_noise

    def get_soft_clusters(self, enc_output):
        n = enc_output.shape[0]
        d = enc_output.shape[1]
        cl = self.clusters.shape[0]

        q = 1.0 / (1.0 + (torch.sum(torch.pow(enc_output.reshape(n, 1, d).repeat(1, cl, 1) - self.clusters, 2),
                                    2) / self.alpha))
        q = q ** ((self.alpha + 1.0) / 2.0)
        probs = (q.t() / (torch.sum(q, 1) + 1e-20)).t()
        return probs

    def get_centers(self):
        return self.clusters.detach().cpu().numpy()

    def set_centers(self, centers):
        print(centers.shape)
        self.cluster_set = True
        self.clusters = torch.tensor(centers, requires_grad=True)

    def forward(self, state, action, mean, log_std):
        input = torch.cat((state, mean), -1)
        input = torch.cat((input, log_std), -1)
        x = F.relu(self.enc_linear1(input))
        x = F.relu(self.enc_linear2(x))
        enc_output = self.enc_linear3(x)

        output_options = self.get_soft_clusters(enc_output)

        inpshape = input.shape
        epsilon = Normal(0, 1).sample(inpshape)
        x2 = input + self.vat_noise * epsilon * torch.abs(input)

        input_noise = self.enc_linear1(x2)
        hidden2_noise = self.enc_linear2(input_noise)
        out_noise = self.enc_linear3(hidden2_noise)
        output_option_noise = self.get_soft_clusters(out_noise)

        dec_1 = F.relu(self.dec_linear1(enc_output))
        dec_2 = F.relu(self.dec_linear2(dec_1))
        dec_output = self.dec_linear3(dec_2)
        return enc_output, output_options, output_option_noise, dec_output, input
