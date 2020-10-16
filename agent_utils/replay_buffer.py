import torch
import random
import numpy as np

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, flag=False):
        batch = random.sample(self.buffer, batch_size)
        if flag:
            batch = self.buffer[0:batch_size]
        state, action, reward, next_state, done = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def clear(self):
        del self.buffer
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


class ReplayBufferWeighted:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, entropy, log_prob, reward, next_state, done, q_value, value, yt):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            state, action, entropy, log_prob, reward, next_state, done, q_value, value, yt)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, flag=False):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        if flag:
            batch = self.buffer[0:batch_size]

        state, action, entropy, log_prob, reward, next_state, done, q_value, value, yt = zip(*batch)

        return state, action, entropy, log_prob, reward, next_state, done, q_value, value, yt

    def clear(self):
        del self.buffer
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)
