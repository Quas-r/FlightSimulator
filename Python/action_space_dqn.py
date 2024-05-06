import random
import numpy as np
import torch


class ActionSpaceDQN(object):

    def __init__(self, dtype=np.float32):
        self.roll_low = np.array([0, -1, 0, 0], dtype=dtype)
        self.roll_high = np.array([0, 1, 0, 0], dtype=dtype)
        self.pitch_low = np.array([0, 0, -1, 0], dtype=dtype)
        self.pitch_high = np.array([0, 0, 1, 0], dtype=dtype)
        self.thrust_low = np.array([-1, 0, 0, 0], dtype=dtype)
        self.thrust_high = np.array([1, 0, 0, 0], dtype=dtype)
        self.yaw_low = np.array([0, 0, 0, -1], dtype=dtype)
        self.yaw_high = np.array([0, 0, 0, 1], dtype=dtype)
        self.no_action = np.array([0, 0, 0, 0], dtype=dtype)

        self.actions = [self.roll_low, self.roll_high,
                        self.pitch_low, self.pitch_high,
                        self.thrust_low, self.thrust_high,
                        self.yaw_low, self.yaw_high, self.no_action]

        self.action_tensor = None

    def __len__(self):
        return len(self.actions)

    def select_random_action(self):

        # selected_action = random.choice(self.actions)
        # return selected_action

        index = random.randint(0, len(self.actions) - 1)
        return index, self.actions[index]

    def update_action_tensor(self, index, device):
        self.action_tensor = torch.tensor([index], device=device, dtype=torch.long)
