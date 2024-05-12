import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class State(object):
    def __init__(self, device):
        self.state = torch.zeros(19, device=device)

    def update_state(self, json_data, device):

        # self.state[0:3] = torch.tensor(
        #     [json_data['playerPlanePositionx'], json_data['playerPlanePositiony'], json_data['playerPlanePositionz']],
        #     device=device)
        # self.state[3:6] = torch.tensor(
        #     [json_data['playerPlaneEulerRotationx'], json_data['playerPlaneEulerRotationy'], json_data['playerPlaneEulerRotationz']],
        #     device=device)
        # self.state[6:9] = torch.tensor(
        #     [json_data['enemyPlanePositionx'], json_data['enemyPlanePositiony'], json_data['enemyPlanePositionz']],
        #     device=device)
        # self.state[9:12] = torch.tensor(
        #     [json_data['enemyPlaneEulerRotationx'], json_data['enemyPlaneEulerRotationy'],  json_data['enemyPlaneEulerRotationz']],
        #     device=device)
        self.state[0:3] = torch.tensor(
            [json_data['relativePositionx'], json_data['relativePositiony'], json_data['relativePositionz']],
            device=device)
        self.state[3:6] = torch.tensor(
            [json_data['playerPlaneEulerRotationx'], json_data['playerPlaneEulerRotationy'], json_data['playerPlaneEulerRotationz']],
            device=device)
        self.state[6:9] = torch.tensor(
            [json_data['enemyPlaneEulerRotationx'], json_data['enemyPlaneEulerRotationy'],  json_data['enemyPlaneEulerRotationz']],
            device=device)
        # self.state[15] = torch.tensor([json_data['playerPlaneForwardVelocity']], device=device)
        # self.state[16] = torch.tensor([json_data['enemyPlaneForwardVelocity']], device=device)
        # self.state[17] = torch.tensor([json_data['playerPlaneGForce']], device=device)
        # self.state[18] = torch.tensor([json_data['enemyPlaneGForce']], device=device)
        self.state[9] = torch.tensor([json_data['playerPlaneForwardVelocity']], device=device)
        self.state[10] = torch.tensor([json_data['enemyPlaneForwardVelocity']], device=device)
        self.state[11] = torch.tensor([json_data['playerPlaneGForce']], device=device)
        self.state[12] = torch.tensor([json_data['enemyPlaneGForce']], device=device)
        self.state[13:16] = torch.tensor([json_data['enemyAngularVelocityx'], json_data['enemyAngularVelocityy'], json_data['enemyAngularVelocityz']],
            device=device)
        self.state[16] = torch.tensor([json_data['enemyThrustValue']], device=device)
        self.state[17] = torch.tensor([json_data['aaAngle']], device=device)
        self.state[18] = torch.tensor([json_data['ataAngle']], device=device)

    def get_state_tensor(self):
        return self.state

    def __len__(self):
        # print(self.state.size(dim=0))
        return self.state.size(dim=0)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.state = torch.clone(self.state)
        result.__dict__.update(self.__dict__)
        return result

    # TODO
    # input verileri olan state normalizasyonu

    @staticmethod
    def apply_min_max_scaling(state):
        numpy_array = state.numpy()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numpy_array)
        return torch.tensor(scaled_data)

    # yuksek ihtimalle bunu kullanacagiz
    @staticmethod
    def apply_standard_scaling(state):
        numpy_array = state.numpy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numpy_array)
        return torch.tensor(scaled_data)

    @staticmethod
    def apply_robust_scaling(state):
        numpy_array = state.numpy()
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numpy_array)
        return torch.tensor(scaled_data)
