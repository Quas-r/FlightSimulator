import torch


class State(object):
    def __init__(self, device):
        self.state = torch.zeros(16, device=device)

    def update_state(self, json_data, device):

        self.state[0:3] = torch.tensor(
            [json_data['playerPlanePositionx'], json_data['playerPlanePositiony'], json_data['playerPlanePositionz']],
            device=device)
        self.state[3:6] = torch.tensor(
            [json_data['playerPlaneEulerRotationx'], json_data['playerPlaneEulerRotationy'], json_data['playerPlaneEulerRotationz']],
            device=device)
        self.state[6:9] = torch.tensor(
            [json_data['enemyPlanePositionx'], json_data['enemyPlanePositiony'], json_data['enemyPlanePositionz']],
            device=device)
        self.state[9:12] = torch.tensor(
            [json_data['enemyPlaneEulerRotationx'], json_data['enemyPlaneEulerRotationy'], json_data['enemyPlaneEulerRotationz']],
            device=device)

        self.state[12] = torch.tensor([json_data['playerPlaneForwardVelocity']], device=device)
        self.state[13] = torch.tensor([json_data['enemyPlaneForwardVelocity']], device=device)
        self.state[14] = torch.tensor([json_data['playerPlaneGForce']], device=device)
        self.state[15] = torch.tensor([json_data['enemyPlaneGForce']], device=device)

    def get_state_tensor(self):
        return self.state

    def __len__(self):
        print(self.state.size(dim=0))
        return self.state.size(dim=0)
