from pynput import keyboard
import socket
import json

HOST = "127.0.0.1"
PORT = 8888

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
    s.bind((HOST, PORT))
    s.listen()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    while True:
        conn, addr = s.accept()
        with conn:
            inp = {"thrust": 0.00,
                   "rollPitch": [0.00, 0.00],
                   "yaw": 0.00,
                   "toggleFlaps": 0}
            while True:
                changed = process_keys_pressed(keys_pressed, inp)
                if changed:
                    data = json.dumps(inp)
                    conn.sendall(bytes(data, "utf-8"))
                    response = conn.recv(9)
                    if response.decode("utf-8") == "ENDGAME":
                        break
                    inp = {"thrust": 0.00,
                           "rollPitch": [0.00, 0.00],
                           "yaw": 0.00,
                           "toggleFlaps": 0}
