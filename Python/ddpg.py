import torch

import torch.nn.functional as F
import torch.optim as optim

from nets import Actor, Critic
from action_space import ActionSpace
from replay_buffer import Transition

INPUT_SIZE = 10
OUTPUT_SIZE = 4
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


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    def __init__(self):

        self.action_space = ActionSpace()

        self.actor = Actor(input_size=INPUT_SIZE, output_size=self.action_space.__len__())
        self.actor_target = Actor(input_size=INPUT_SIZE, output_size=self.action_space.__len__())
        self.critic = Critic(input_size=INPUT_SIZE, output_size=self.action_space.__len__())
        self.critic_target = Critic(input_size=INPUT_SIZE, output_size=self.action_space.__len__())

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=LR, amsgrad=True)
        self.critic_optimizer = optim.AdamW(self.actor.parameters(), lr=LR, amsgrad=True)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def calculate_action(self, state, action_noise=None):

        x = state.to(device)

        self.actor.eval()
        output = self.actor(x)
        self.actor.train()
        output = output.data

        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            output += noise

            output = torch.clamp(output, self.action_space.low, self.action_space.high)

        return output

    def optimize_model(self, buffer):

        if len(buffer) < BATCH_SIZE:
            return
        transitions = buffer.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * GAMMA * next_state_action_values

        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.actor_target, self.actor, TAU)
        soft_update(self.critic_target, self.critic, TAU)

        return value_loss.item(), policy_loss.item()
