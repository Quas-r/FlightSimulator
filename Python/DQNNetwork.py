import random
import torch
import math
from torch import nn
from torch.cuda import is_available
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from typing import Deque, List

#TODO -> Plotting

INPUT_SIZE = 10 #TODO
OUTPUT_SIZE = 4 #TODO
BUFFER_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.98
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = Deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        self.input_layer = nn.Linear(input_size, 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)

class DQNNetwork(object):
    def __init__(self):
        self.policy_net = DQNModel(INPUT_SIZE, OUTPUT_SIZE).to(device)
        self.target_net = DQNModel(INPUT_SIZE, OUTPUT_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

        self.steps_done = 0

        def select_action(self, state):
            probability = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
            self.steps_done += 1
            if probability > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return 0 #TODO

        
