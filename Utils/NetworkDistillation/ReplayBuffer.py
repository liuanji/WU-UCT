import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size = 1e6, device = None):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

        self.device = device

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size = batch_size)
        state_batch, policy_batch, value_batch = [], [], []

        for i in ind:
            state, policy, value = self.storage[i]
            state_batch.append(np.array(state, copy = False))
            policy_batch.append(np.array(policy, copy = False))
            value_batch.append(value)

        state_batch = torch.from_numpy(np.array(state_batch, dtype = np.float32))
        policy_batch = torch.from_numpy(np.array(policy_batch, dtype = np.float32))
        value_batch = torch.from_numpy(np.array(value_batch, dtype = np.float32))

        if self.device is not None:
            state_batch = state_batch.to(self.device)
            policy_batch = policy_batch.to(self.device)
            value_batch = value_batch.to(self.device).unsqueeze(-1)

        return state_batch, policy_batch, value_batch
