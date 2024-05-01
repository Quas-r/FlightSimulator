import torch


class State(object):
    def __init__(self, device):
        self.state = torch.zeros(18, device=device)

    def update_state(self, json_data, device):

        self.state[0:3] = torch.tensor(
            [json_data['planePositionx'], json_data['planePositiony'], json_data['planePositionz']],
            device=device)
        self.state[3:6] = torch.tensor(
            [json_data['planeEulerRotationx'], json_data['planeEulerRotationy'], json_data['planeEulerRotationz']],
            device=device)
        self.state[6:9] = torch.tensor(
            [json_data['cubePositionx'], json_data['cubePositiony'], json_data['cubePositionz']],
            device=device)
        self.state[9:12] = torch.tensor(
            [json_data['cubeEulerRotationx'], json_data['cubeEulerRotationy'], json_data['cubeEulerRotationz']],
            device=device)

        self.state[12] = torch.tensor([json_data['planeVelocity']], device=device)
        self.state[13] = torch.tensor([json_data['cubeVelocity']], device=device)
        self.state[14] = torch.tensor([json_data['planeGForce']], device=device)
        self.state[15] = torch.tensor([json_data['cubeGForce']], device=device)

    def get_state_tensor(self):
        return self.state

    def __len__(self):
        return self.state
