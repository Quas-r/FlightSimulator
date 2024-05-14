import random
import numpy as np
import torch
import math
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque


# TODO -> Plotting

BUFFER_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.98
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.hidden_layer2 = nn.Linear(256,256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):

        # TODO
        # Ara katmanda egitimi hizlandirmak icin relu
        # hatta leaky relu kullanabiliriz negatif girdiler kacmamis olur
        # ya da tanh kullanabiliriz

        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))

        # cikis katmaninda dogrusal olmasi daha iyi
        return self.output_layer(x)


class DQNNetwork(object):
    def __init__(self, input_size, output_size, device):
        self.policy_net = DQNModel(input_size, output_size).to(device)
        self.target_net = DQNModel(input_size, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

        # TODO
        # Agirliklar normalize edilecek mi
        # l2 (torch.nn.utils.spectral_norm da kullanabiliriz)
        # self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-5)

        self.buffer = ReplayBuffer(capacity=BUFFER_SIZE)

        self.steps_done = 0

    def select_action(self, state, action_space, device):
        probability = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if probability > eps_threshold:
            with torch.no_grad():
                # TODO
                # Float yerine int mantıklı olabilir mi?
                output = self.policy_net(state)
                max_value, max_index = output.max(0)
                action_tensor = torch.tensor([action_space.actions[max_index]], device=device, dtype=torch.float32)
                return action_tensor, max_index
        else:
            index, action = action_space.select_random_action()
            action_np = np.array([action])
            action_tensor = torch.tensor(action_np, device=device, dtype=torch.long)
            return action_tensor, index

    def optimize_model(self, device):

        if len(self.buffer) < BATCH_SIZE:
            return
        # batch = self.buffer.sample(batch_size=BATCH_SIZE)
        transitions = self.buffer.sample(batch_size=BATCH_SIZE)
        batch = Transition(*zip(*transitions))
    
        non_final_mask = torch.tensor([transition.next_state is not None for transition in transitions],
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.stack(
            [state.get_state_tensor() for state in batch.next_state if state is not None], dim=0)

        state_batch = torch.stack([state.get_state_tensor() for state in batch.state], dim=0)
        action_batch = torch.stack(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, filename="D:/FlightSimulator/FlightSimulator/Models/256_256_1e-3.pth"):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load_model(self, filename="D:/FlightSimulator/FlightSimulator/Models/256_256_1e-3.pth"):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_net.eval()
        self.policy_net.train()
