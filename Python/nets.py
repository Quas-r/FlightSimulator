import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):

    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()

        self.input_layer = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)

        self.hidden_layer = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)

        self.output_layer = nn.Linear(128, output_size)

        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):

        x = self.input_layer(inputs)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.hidden_layer(x)
        x = self.ln2(x)
        x = F.relu(x)

        output = torch.tanh(self.output_layer(x))
        return output


class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)

        self.hidden_layer = nn.Linear(128 + output_size, 128)
        self.ln2 = nn.LayerNorm(128)

        self.output_layer = nn.Linear(128, 1)

        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.output_layer.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.output_layer.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):

        x = self.input_layer(inputs)
        x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.hidden_layer(x)
        x = self.ln2(x)
        x = F.relu(x)

        output = self.output_layer(x)
        return output
