import torch
import numpy as np
from misc.torch_utils import combined_shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBufferSOC(object):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, num_tasks, size):
        self.state_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.option_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.next_state_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.task_num_buf = np.zeros(size, dtype=np.int32)
        self.ptr, self.size, self.num_tasks, self.max_size = 0, 0, num_tasks, size

    def store(self, state, option, action, reward, next_state, done, task_num):
        self.state_buf[self.ptr] = state
        self.option_buf[self.ptr] = option
        self.next_state_buf[self.ptr] = next_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.task_num_buf[self.ptr] = task_num
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        batches = []
        for i in range(self.num_tasks):
            specific_idx = np.argwhere(self.task_num_buf == i).flatten()
            idxs = np.random.choice(specific_idx, batch_size)
            batch = dict(
                state=self.state_buf[idxs],
                option=self.option_buf[idxs],
                action=self.action_buf[idxs],
                reward=self.reward_buf[idxs],
                next_state=self.next_state_buf[idxs],
                done=self.done_buf[idxs],
                task_num=self.task_num_buf[idxs])
            batches.append(batch)

        return [{k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in task_batch.items()} for task_batch in batches]
