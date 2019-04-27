from collections import namedtuple
import random

Transition = namedtuple("Transition",('state','action','next_state','reward'))

class ReplayMemory:
    def __init__(self, max_size):
        self.memory = []
        self.max_size = max_size
        self.position = 0

    def add_sample(self, *args):
        if(len(self.memory) < self.max_size):
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.max_size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, b):
        return self.memory[b]

