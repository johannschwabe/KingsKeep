import random


def get_class_name():
    return 'RandomPlayer'


class RandomPlayer:
    def __init__(self):
        self.name = "Random Player"
        self.uzh_shortname = "pplayer"

    def move_sheep(self, p_num, p_state, p_time_remaining, field):
        return random.randint(0, 4) - 2, p_state

    def move_wolf(self, p_num, p_state, p_time_remaining, field):
        return random.randint(0, 4) - 2, p_state
