import random
import torch
from config import *


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


class RLPlayer:
    def __init__(self, sheep_policy_net, wolf_policy_net, strategy, device):
        self.name = "RL Player"
        self.sheep_policy_net = sheep_policy_net
        self.wolf_policy_net = wolf_policy_net
        self.current_step = 0
        self.strategy = strategy
        self.device = device

    @staticmethod
    def convert_field_to_state(field, device):
        nested_state = [[FIELD_ELEMENTS_MAPPER[j] for j in i] for i in field]
        flat_state = [item for sublist in nested_state for item in sublist]
        return torch.tensor([flat_state]).to(device)

    def compute_move(self, field, is_sheep_move):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            # explore
            action = random.randrange(5)
            move = torch.tensor([action]).to(self.device)
        else:
            # exploit
            state = self.convert_field_to_state(field=field, device=self.device)
            policy_net = self.sheep_policy_net if is_sheep_move else self.wolf_policy_net
            with torch.no_grad():
                move = policy_net(state).argmax(dim=1).to(self.device)

        return int(move) - 2

    def move_sheep(self, p_num, field):
        return self.compute_move(field=field, is_sheep_move=True)

    def move_wolf(self, p_num, field):
        return self.compute_move(field=field, is_sheep_move=False)
