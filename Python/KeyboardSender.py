import socket
import json
from copy import copy

import torch
from DQNNetwork import DQNNetwork
from action_space_dqn import ActionSpaceDQN
from plot_actions import plot_action_histogram
from plot_reward_2 import plot_reward_of_all_time

from plot_rewards import plot_rewards
from state import State

HOST = "127.0.0.1"
PORT = 8888
TAU = 0.005

changed = False

keys_pressed = {"a": False,
                "d": False,
                "w": False,
                "s": False,
                "q": False,
                "e": False,
                "shift": False,
                "ctrl": False}


def apply_key(key_value, apply_value: bool):
    global keys_pressed
    if key_value in keys_pressed:
        keys_pressed[key_value] = apply_value


def on_press(key):
    try:
        apply_key(key.char, True)
    except AttributeError:
        apply_key(key.name, True)


def on_release(key):
    try:
        apply_key(key.char, False)
    except AttributeError:
        apply_key(key.name, False)


def process_keys_pressed(keys_pressed, inp) -> bool:
    changed = True
    if keys_pressed["a"]:
        inp["rollPitch"][0] = -1.00 
    elif keys_pressed["d"]:
        inp["rollPitch"][0] = 1.00 
    elif keys_pressed["w"]:
        inp["rollPitch"][1] = 1.00 
    elif keys_pressed["s"]:
        inp["rollPitch"][1] = -1.00 
    elif keys_pressed["q"]:
        inp["yaw"] = -1.00 
    elif keys_pressed["e"]:
        inp["yaw"] = 1.00 
    elif keys_pressed["shift"]:
        inp["thrust"] = 1.00 
    elif keys_pressed["ctrl"]:
        inp["thrust"] = -1.00 
    else:
        changed = False
    return changed


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    try:
        s.bind((HOST, PORT))
        s.listen()
        print("Server started...")
    except socket.error as e:
        print(f"Server startup error: {e}")

    rewards_all_time = []

    while True:
        print("Waiting for connection...")
        conn, addr = s.accept()
        print("Connection established!")
        with conn:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            rewards = []
            actions_index = []

            actions = ActionSpaceDQN()

            state = State(device=device)
            observation = State(device=device)

            input_size = state.__len__()
            output_size = actions.__len__()

            model = DQNNetwork(input_size=input_size, output_size=output_size, device=device)
            try:
                model.load_model()
            except FileNotFoundError:
                print("No already existing model detected.")

            print(model)

            while True:

                observation = State(device=device)

                action, index = model.select_action(state.get_state_tensor(), actions, device=device)
                action_index_tensor = torch.tensor([index], device=device)
                actions_index.append(index)

                inp = {
                    "thrust": action[0][0].item(),
                    "rollPitch": [action[0][1].item(), action[0][2].item()],
                    "yaw": action[0][3].item(),
                    "toggleFlaps": 0
                }

                received_data = conn.recv(1024)
                received_data = received_data.decode('utf-8')
                if not received_data:
                    # plot_rewards(rewards)
                    # plot_action_histogram(actions_index, actions)
                    plot_reward_of_all_time(rewards_all_time)
                    model.save_model()
                    print(f"Steps done: {model.steps_done}")
                    print("Connection is being terminated...")
                    break
                data_dict = json.loads(received_data)
                # print("Received:", received_data)

                # TODO
                # Veri alindiktan sonra normalize edilmis degerler tutulacak statede
                # bir kere yapalim bu degistirmeyi her input verildiginde degil
                # state.apply_standard_scaling(data_dict)

                data_to_send = json.dumps(inp)
                conn.sendall(bytes(data_to_send, "utf-8"))
                # print("Sent:", data_to_send)

                observation.update_state(data_dict, device=device)
                reward = data_dict["reward"]
                # print("Reward: ", reward)

                if data_dict["endGame"] == "CONGAME":
                    terminated = False
                else:
                    terminated = True

                reward = torch.tensor([reward], device=device)
                rewards.append(reward)
                rewards_all_time.append(reward)

                if terminated:
                    next_state = None
                else:
                    next_state = observation

                model.buffer.push(state, action_index_tensor, next_state, reward)

                state = copy(next_state)

                model.optimize_model(device=device)

                target_net_state_dict = model.target_net.state_dict()
                policy_net_state_dict = model.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                model.target_net.load_state_dict(target_net_state_dict)

                if next_state is None:

                    # TODO
                    # hangi diagramlar cizdirilecek
                    # plot_rewards(rewards)
                    # plot_action_histogram(actions_index, actions)

                    model.save_model()
                    print(f"Steps done: {model.steps_done}")
                    print("Connection is being terminated...")
                    break

        # plot_reward_of_all_time(rewards_all_time)
