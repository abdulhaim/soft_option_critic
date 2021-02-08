from collections import deque
import numpy as np
import random

class ReplayBufferSAC(object):
    def __init__(self, capacity):
        self.max_size = capacity
        self.buffer = deque(maxlen=capacity)
        self.ptr = 0
        self.size = 0

    def store(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def store_mer(self, state, action, reward, next_state, done):

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        if self.size >= self.max_size:
            p = random.randint(0, self.ptr)
            if p < self.max_size:
                state = np.expand_dims(state, 0)
                next_state = np.expand_dims(next_state, 0)
                self.buffer[p] = (state, action, reward, next_state, done)

        else:
            self.buffer.append((state, action, reward, next_state, done))

        self.size = min(self.size + 1, self.max_size)
        self.ptr += 1
#
    def sample_batch(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


# import torch
# import random
# from misc.torch_utils import combined_shape
# import numpy as np
#
#
# class ReplayBufferSAC(object):
#     """
#     Replay Buffer for Soft Actor Critic
#     """
#
#     def __init__(self, obs_dim, act_dim, size):
#         self.state_buf = np.zeros(combined_shape(size, 84, 84, 4), dtype=np.float32)
#         self.next_state_buf = np.zeros(combined_shape(size, 84, 84, 4), dtype=np.float32)
#         self.action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
#         self.reward_buf = np.zeros(size, dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#         self.ptr, self.size, self.max_size = 0, 0, size
#
#     def store(self, obs, act, rew, next_obs, done):
#         self.state_buf[self.ptr] = obs
#         self.next_state_buf[self.ptr] = next_obs
#         self.action_buf[self.ptr] = act
#         self.reward_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def store_mer(self, obs, act, rew, next_obs, done):
#         if self.size >= self.max_size:
#             p = random.randint(0, self.ptr)
#             if p < self.max_size:
#                 self.state_buf[p] = obs
#                 self.next_state_buf[p] = next_obs
#                 self.action_buf[p] = act
#                 self.reward_buf[p] = rew
#                 self.done_buf[p] = done
#
#         else:
#             self.state_buf[self.ptr] = obs
#             self.next_state_buf[self.ptr] = next_obs
#             self.action_buf[self.ptr] = act
#             self.reward_buf[self.ptr] = rew
#             self.done_buf[self.ptr] = done
#
#         self.size = min(self.size + 1, self.max_size)
#         self.ptr += 1
#
#     def sample_batch(self, batch_size=32):
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         batch = dict(
#             state=self.state_buf[idxs],
#             next_state=self.next_state_buf[idxs],
#             action=self.action_buf[idxs],
#             reward=self.reward_buf[idxs],
#             done=self.done_buf[idxs])
#         return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
