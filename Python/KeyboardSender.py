from pynput import keyboard
import socket
import json

import torch
from DQNNetwork import DQNNetwork
from action_space_dqn import ActionSpaceDQN
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

    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # listener.start()

    try:
        s.bind((HOST, PORT))
        s.listen()
        print("Server başlatıldı, bağlantı bekleniyor...")
    except socket.error as e:
        print(f"Sunucu başlatma hatası: {e}")

    while True:
        conn, addr = s.accept()
        with conn:

            inp = {"thrust": 0.00,
                   "rollPitch": [0.00, 0.00],
                   "yaw": 0.00,
                   "toggleFlaps": 0}

            # /////////////////////

            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # model olusturma
            state = State(device=device)  # TODO
            observation = State(device=device)  # TODO

            actions = ActionSpaceDQN()

            input_size = state.__len__()
            output_size = actions.__len__()

            model = DQNNetwork(input_size=input_size, output_size=output_size, device=device)

            print(model)

            # TODO
            # Kayıtlı model var mı kontrol edilecek
            # Model ağırlıkları yüklenecek
            # model.load_model()

            data_to_send = json.dumps(inp)
            conn.sendall(bytes(data_to_send, "utf-8"))

            received_data = conn.recv(1024)
            received_data = received_data.decode('utf-8')
            print(received_data)
            data_dict = json.loads(received_data)
            state.update_state(data_dict, device=device)

            counter = 1
            while True:

                action, index = model.select_action(state.get_state_tensor(), actions, device=device)
                action_index_tensor = torch.tensor([index], device=device)

                inp = {
                    "thrust": action[0][0].item(),
                    "rollPitch": [action[0][1].item(), action[0][2].item()],
                    "yaw": action[0][3].item(),
                    "toggleFlaps": 0
                }

                print(inp)

                # TODO
                # Unity tarafına veri gönderilecek action
                data_to_send = json.dumps(inp)
                conn.sendall(bytes(data_to_send, "utf-8"))
                print(data_to_send)

                print("VERİ GÖNDERİLDİ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                # TODO
                # Unity tarafındna veri alınacak
                received_data = conn.recv(1024)
                received_data = received_data.decode('utf-8')
                data_dict = json.loads(received_data)

                observation.update_state(data_dict, device=device)
                reward = data_dict["reward"]
                if data_dict["endGame"] == "CONGAME":
                    terminated = False
                else:
                    terminated = True
                print(data_dict["endGame"])

                reward = torch.tensor([reward], device=device)

                if terminated:
                    next_state = None
                else:
                    next_state = observation

                model.buffer.push(state, action_index_tensor, next_state, reward)

                state = next_state

                model.optimize_model(device=device)

                target_net_state_dict = model.target_net.state_dict()
                policy_net_state_dict = model.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                model.target_net.load_state_dict(target_net_state_dict)

                if next_state is None:
                    # TODO
                    # Modelin ağırlıkları kaydedilecek
                    # model.save_model()
                    break

                # ////////////////////////////////

                # changed = process_keys_pressed(keys_pressed, inp)
                # if changed:
                #     data = json.dumps(inp)
                #     conn.sendall(bytes(data, "utf-8"))
                #     response = conn.recv(9)
                #     response = response.decode('utf-8')
                #     data_dict = json.loads(response)
                #     if data_dict["endgame"] == "EDNGAME":
                #         break
                #     inp = {"thrust": 0.00,
                #            "rollPitch": [0.00, 0.00],
                #            "yaw": 0.00,
                #            "toggleFlaps": 0}

                # data_to_send = json.dumps(inp)
                # conn.sendall(bytes(data_to_send, "utf-8"))
                # received_data = conn.recv(1024)
                #
                # if not received_data:
                #     print("Bağlantı kapatıldı.")
                #     break
                #
                # received_data = received_data.decode('utf-8')
                # data_dict = json.loads(received_data)
                #
                # state = State()
                # next_state = state.update_state(data_dict)
                # print(next_state)
                # reward = data_dict["reward"]
                # print(reward)
                #
                #
                # if data_dict["endGame"] == "EDNGAME":
                #     print("Oyunun sonlandırılması istendi.")
