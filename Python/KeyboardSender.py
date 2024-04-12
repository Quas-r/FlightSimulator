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


def on_press(key):
    global keys_pressed
    try:
        if key.char == "a":
            keys_pressed["a"] = True
        elif key.char == "d":
            keys_pressed["d"] = True
        elif key.char == "w":
            keys_pressed["w"] = True
        elif key.char == "s":
            keys_pressed["s"] = True
        elif key.char == "q":
            keys_pressed["q"] = True
        elif key.char == "e":
            keys_pressed["e"] = True
    except AttributeError:
        if key.name == "shift":
            keys_pressed["shift"] = True
        elif key.name == "ctrl":
            keys_pressed["ctrl"] = True

def on_release(key):
    global keys_pressed
    try:
        if key.char == "a":
            keys_pressed["a"] = False
        elif key.char == "d":
            keys_pressed["d"] = False
        elif key.char == "w":
            keys_pressed["w"] = False
        elif key.char == "s":
            keys_pressed["s"] = False
        elif key.char == "q":
            keys_pressed["q"] = False
        elif key.char == "e":
            keys_pressed["e"] = False
    except AttributeError:
        if key.name == "shift":
            keys_pressed["shift"] = False
        elif key.name == "ctrl":
            keys_pressed["ctrl"] = False

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
    conn, addr = s.accept()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
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
                conn.recv(9)
                inp = {"thrust": 0.00,
                       "rollPitch": [0.00, 0.00],
                       "yaw": 0.00,
                       "toggleFlaps": 0}
    listener.join()
