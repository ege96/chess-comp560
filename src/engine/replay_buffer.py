import numpy as np
import random
from collections import deque
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=10000):
        """
        Prioritized Experience Replay buffer.
        alpha: how much prioritization to use (0 = uniform sampling, 1 = full prioritization)
        beta: correction for importance sampling bias (0.4 -> 1.0 annealed over training)
        """
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.frame = 0

    def update_beta(self, frame_idx):
        """Update beta parameter for importance sampling bias correction."""
        self.frame = frame_idx
        self.beta = min(self.beta_end, self.beta_start + frame_idx * (self.beta_end - self.beta_start) / self.beta_frames)

    def push(self, state, reward, next_state, done, policy_target, priority=None):
        """Add experience with optional priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        if priority is None:
            # If no priority provided, use max priority
            if len(self.buffer) == 0:
                max_priority = 1.0
            else:
                max_priority = np.max(self.priorities[:len(self.buffer)])
            priority = max_priority
        
        self.buffer[self.position] = (state, reward, next_state, done, policy_target)
        self.priorities[self.position] = priority**self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch_size experiences with prioritization."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        # Add small constant to avoid zero priority
        prios = prios + 1e-5
        
        # Normalized priorities for sampling
        prios_sum = np.sum(prios)
        if prios_sum == 0:
            # If all priorities are zero, use uniform sampling
            probs = np.ones_like(prios) / len(prios)
        else:
            probs = prios / prios_sum
        
        # Check for NaN and fix (should be redundant with above checks, but safety first)
        if np.isnan(probs).any():
            probs = np.ones_like(probs) / len(probs)
        
        # Sample according to priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()  # Normalize
        
        # Unpack the batch
        state, reward, next_state, done, policy_target = zip(*samples)
        
        return (
            state, reward, next_state, done, policy_target,
            indices, torch.FloatTensor(weights)
        )

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority**self.alpha

    def __len__(self):
        return len(self.buffer)


# Keep ReplayBuffer for backward compatibility and testing
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