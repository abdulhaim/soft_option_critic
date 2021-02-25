from collections import deque
import numpy as np
import random

class ReplayBufferSOC(object):
    def __init__(self, capacity):
        self.max_size = capacity
        self.buffer = deque(maxlen=capacity)
        self.ptr = 0
        self.size = 0

    def store(self, state, option, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, option, action, reward, next_state, done))

    def store_mer(self, state, option, action, reward, next_state, done):

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        if self.size >= self.max_size:
            p = random.randint(0, self.ptr)
            if p < self.max_size:
                state = np.expand_dims(state, 0)
                next_state = np.expand_dims(next_state, 0)
                self.buffer[p] = (state, option, action, reward, next_state, done)

        else:
            self.buffer.append((state, option, action, reward, next_state, done))

        self.size = min(self.size + 1, self.max_size)
        self.ptr += 1
#
    def sample_batch(self, batch_size):
        state, option, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), option, action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
