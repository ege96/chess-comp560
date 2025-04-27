import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, reward, next_state, done = map(list, zip(*batch))
        return state, reward, next_state, done

    def __len__(self):
        return len(self.buffer) 