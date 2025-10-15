from collections import deque
import random


class ReplayMemory():
    def __init__(self, maximum, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
