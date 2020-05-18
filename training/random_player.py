import random


def get_class_name():
    return 'RandomPlayer'


class RandomPlayer:
    def __init__(self):
        self.name = "Random Player"
        self.uzh_shortname = "pplayer"

    def move_sheep(self, p_num, field):
        return random.randint(0, 4) - 2

    def move_wolf(self, p_num, field):
        return random.randint(0, 4) - 2
