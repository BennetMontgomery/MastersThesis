import random

class ReplayManager:
    def __init__(self, replay_cap):
        self.capacity = replay_cap
        self.memory = []
        self.counter = 0

    def add_mem(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.counter % self.capacity] = experience

        self.counter += 1

    def sample_batch(self, batch_size):
        if len(self.memory) > batch_size:
            return random.sample(self.memory, batch_size)
        else:
            raise ValueError(f"[!!] Replay Buffer queried before {batch_size} memories accumulated")