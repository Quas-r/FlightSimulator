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

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

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

        # TODO
        # Ara katmanda egitimi hizlandirmak icin relu
        # hatta leaky relu kullanabiliriz negatif girdiler kacmamis olur
        # ya da tanh kullanabiliriz

        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))

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

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()

        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, filename="checkpoint.pth"):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load_model(self, filename="checkpoint.pth"):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_net.eval()
        self.policy_net.train()

# def train_model(state, model, actions):  # bu kullanilmayabilir
#
#     action = model.select_action(state, actions)
#
#     observation = None
#     reward = None
#     terminated = None
#
#     reward = torch.tensor([reward], device=device)
#
#     if terminated:
#         next_state = None
#     else:
#         next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
#
#     model.buffer.push(state, action, next_state, reward)
#
#     state = next_state
#
#     model.optimize_model()
#
#     target_net_state_dict = model.target_net.state_dict()
#     policy_net_state_dict = model.policy_net.state_dict()
#     for key in policy_net_state_dict:
#         target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
#     model.target_net.load_state_dict(target_net_state_dict)
#
#     return state
#
#
# def calculate_degrees(civilian_position, civilian_vector, enemy_position, enemy_vector):
#
#     los = civilian_position - enemy_position
#
#     distance = np.linalg.norm(los)
#
#     cos_angle_aa = np.dot(civilian_vector, los) / (np.linalg.norm(civilian_vector) * np.linalg.norm(los))
#     aa_angle = np.degrees(np.arccos(np.clip(cos_angle_aa, -1.0, 1.0)))
#
#     cos_angle_ata = np.dot(enemy_vector, -los) / (np.linalg.norm(enemy_vector) * np.linalg.norm(los))
#     ata_angle = np.degrees(np.arccos(np.clip(cos_angle_ata, -1.0, 1.0)))
#
#     return aa_angle, ata_angle, distance
#
#
# def calculate_reward(aa, ata, distance, velocity, g_force, enemy_position):
#
#     reward = 0
#
#     high_altitude_speed = 925  # km/s
#     low_and_medium_altitude_speed = (800, 900)  # km/s
#     high_altitude_limit = 12200  # m
#
#     max_g_force = 7
#
#     aim_120_sidewinder_low_distance = 1000  # km
#     aim_120_sidewinder_high_distance = 34500  # km
#
#     if distance < 0.1:
#         reward += -10
#     elif aim_120_sidewinder_low_distance <= distance <= aim_120_sidewinder_high_distance:
#         reward += 10
#         if abs(aa) < 0 and abs(ata) < 0:
#             reward += 10
#     elif aim_120_sidewinder_high_distance < distance:
#         if abs(aa) < 60 and abs(ata) < 30:
#             reward += 2
#         elif abs(ata) > 120 and abs(aa) > 150:
#             reward += -2
#
#     if (enemy_position[1] < high_altitude_limit and
#             low_and_medium_altitude_speed[0] < velocity < low_and_medium_altitude_speed[1]):
#         reward += 1
#     elif enemy_position[1] >= high_altitude_limit and high_altitude_speed == velocity:
#         reward += 1
#     else:
#         reward -= 1
#
#     if g_force <= max_g_force:
#         reward += 5
#
#     return reward
