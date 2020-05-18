from config import *
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F


FIELD_ELEMENTS = [
    CELL_EMPTY,
    CELL_SHEEP_1,
    CELL_SHEEP_1_d,
    CELL_WOLF_1,
    CELL_SHEEP_2,
    CELL_SHEEP_2_d,
    CELL_WOLF_2,
    CELL_GRASS,
    CELL_RHUBARB,
    CELL_FENCE,
]

FIELD_ELEMENTS_MAPPER = {FIELD_ELEMENTS[i]: float(i) for i in list(range(len(FIELD_ELEMENTS)))}


def get_class_name():
    return 'RLPlayer'


class DQN(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()

        self.fc1 = nn.Linear(in_features=n_inputs, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=5)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class RLPlayer:
    def __init__(self):
        self.name = "RL Player"
        self.uzh_shortname = "rlplayer"

    @staticmethod
    def convert_field_to_state(field, device):
        nested_state = [[FIELD_ELEMENTS_MAPPER[j] for j in i] for i in field]
        flat_state = [item for sublist in nested_state for item in sublist]
        return torch.tensor([flat_state]).to(device)

    def get_device(self):
        return torch.device("cpu")

    def get_sheep_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('rlplayer_sheep_model.pt')
        model = DQN(n_inputs=15*19)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model

    def get_wolf_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('rlplayer_wolf_model.pt')
        model = DQN(n_inputs=15*19)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model

    def move_sheep(self, p_num, p_state, p_time_remaining, field):
        if 'sheep_model' not in p_state:
            p_state['sheep_model'] = self.get_sheep_model()

        if 'device' not in p_state:
            p_state['device'] = self.get_device()

        state = self.convert_field_to_state(field=field, device=p_state['device'])
        with torch.no_grad():
            move = p_state['sheep_model'](state).argmax(dim=1).to(p_state['device'])
        return int(move) - 2, p_state

    def move_wolf(self, p_num, p_state, p_time_remaining, field):
        if 'wolf_model' not in p_state:
            p_state['wolf_model'] = self.get_wolf_model()

        if 'device' not in p_state:
            p_state['device'] = self.get_device()

        state = self.convert_field_to_state(field=field, device=p_state['device'])
        with torch.no_grad():
            move = p_state['wolf_model'](state).argmax(dim=1).to(p_state['device'])
        return int(move) - 2, p_state
