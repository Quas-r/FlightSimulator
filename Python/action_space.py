import numpy as np


class ActionSpace:
    def __init__(self, dtype=np.float32):
        self.low = np.array([-1, -1, 0], dtype=dtype)
        self.high = np.array([1, 1, 1], dtype=dtype)

    def __len__(self):
        return len(self.low)
