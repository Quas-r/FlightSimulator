import numpy as np
import torch
from ddpg import DDPG
from replay_buffer import ReplayBuffer
from state import State
from noise import OrnsteinUhlenbeckActionNoise
from sklearn.preprocessing import StandardScaler

BUFFER_SIZE = 10000
BATCH_SIZE = 128
REWARD_THRESHOLD = 100  # bu degisebilir
NOISE_STDDEV = 0.2


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train_ddpg_model():

    # listenerdan veri gelince bu kod işleyecek

    # TODO
    # seed konulacak mi torch, np ve random icin ve cuda icin

    agent = DDPG()
    buffer = ReplayBuffer(capacity=BUFFER_SIZE)

    # TODO
    # noise yerine reward normalization da yapilabilir
    number_of_actions = agent.action_space.__len__()
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(number_of_actions),
                                            sigma=float(NOISE_STDDEV)*np.ones(number_of_actions))

    # burda ilk state alinacak
    state = State()
    normalization_state(state)
    while True:

        ou_noise.reset()

        action = agent.train_model(state)

        # bunlar unity tarafindan gelecek
        next_state = None
        reward = None
        done = None

        mask = torch.Tensor([done]).to(device)
        reward = torch.Tensor([reward]).to(device)
        next_state = torch.Tensor([next_state]).to(device)

        buffer.push(state, action, mask, next_state, reward)

        state = next_state

        value_loss, policy_loss = agent.optimize_model(buffer)


def normalization_state(state):

    scaler = StandardScaler()

    position_normalized = scaler.fit_transform(state.enemy_position.reshape(-1, 1))
    rotation_normalized = scaler.transform(state.enemy_rotation.reshape(-1, 1))
    velocity_normalized = scaler.transform(state.enemy_velocity.reshape(-1, 1))
    g_force_normalized = scaler.transform(state.enemy_g_force.reshape(-1, 1))

    # TODO

    return position_normalized, rotation_normalized, velocity_normalized, g_force_normalized
