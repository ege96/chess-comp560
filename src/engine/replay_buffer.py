import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done, policy_target):
        self.buffer.append((state, reward, next_state, done, policy_target))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, reward, next_state, done, policy_target = map(list, zip(*batch))
        return state, reward, next_state, done, policy_target

    def __len__(self):
        return len(self.buffer) 