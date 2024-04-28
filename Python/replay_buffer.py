import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)