import random
import numpy as np
import torch
import math
from torch import nn
from torch.cuda import is_available
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
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
                return torch.tensor([[generate_random_action()]], device=device, dtype=torch.long)

        def optimize_model():

            if len(self.buffer) < BATCH_SIZE:
                return
            transitions = self.buffer.sample(batch_size=BATCH_SIZE)
            batch = Transition(*zip(*transitions))

        



def generate_random_action():

    random_roll = random.uniform(-1, 1)
    random_pitch = random.uniform(-1, 1)
    random_roll_pitch = (random_roll, random_pitch)
    random_thrust = random.uniform(0, 1)
    random_yaw = random.uniform(-1, 1)

    return random_roll_pitch, random_thrust, random_yaw


def calculate_degrees(civilian_position, civilian_vector, enemy_position, enemy_vector):

    los = civilian_position - enemy_position

    distance = np.linalg.norm(los)

    cos_angle_aa = np.dot(civilian_vector, los) / (np.linalg.norm(civilian_vector) * np.linalg.norm(los))
    aa_angle = np.degrees(np.arccos(np.clip(cos_angle_aa, -1.0, 1.0)))

    cos_angle_ata = np.dot(enemy_vector, -los) / (np.linalg.norm(enemy_vector) * np.linalg.norm(los))
    ata_angle = np.degrees(np.arccos(np.clip(cos_angle_ata, -1.0, 1.0)))

    return aa_angle, ata_angle, distance


def calculate_reward(aa, ata, distance, velocity, g_force, enemy_position):

    reward = 0

    high_altitude_speed = 925  # km/s
    low_and_medium_altitude_speed = (800, 900)  # km/s
    high_altitude_limit = 12200  # m

    max_g_force = 7

    aim_120_sidewinder_low_distance = 1000  # km
    aim_120_sidewinder_high_distance = 34500  # km

    if distance < 0.1:
        reward += -10
    elif aim_120_sidewinder_low_distance <= distance <= aim_120_sidewinder_high_distance:
        reward += 10
        if abs(aa) < 0 and abs(ata) < 0:
            reward += 10
    elif aim_120_sidewinder_high_distance < distance:
        if abs(aa) < 60 and abs(ata) < 30:
            reward += 2
        elif abs(ata) > 120 and abs(aa) > 150:
            reward += -2

    if (enemy_position[1] < high_altitude_limit and
            low_and_medium_altitude_speed[0] < velocity < low_and_medium_altitude_speed[1]):
        reward += 1
    elif enemy_position[1] >= high_altitude_limit and high_altitude_speed == velocity:
        reward += 1
    else:
        reward -= 1

    if g_force <= max_g_force:
        reward += 5

    return reward
