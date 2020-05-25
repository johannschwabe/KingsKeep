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

        self.fc1 = nn.Linear(in_features=n_inputs, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=512)
        self.fc6 = nn.Linear(in_features=512, out_features=512)
        self.fc7 = nn.Linear(in_features=512, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=5)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = F.relu(self.fc5(t))
        t = F.relu(self.fc6(t))
        t = F.relu(self.fc7(t))
        t = self.out(t)
        return t




class KingsKeep:
    def __init__(self):
        self.name = "KingsFeet"
        self.uzh_shortname = "joschwa"


    @staticmethod
    def convert_field_to_state_sheep(field, device):
        sheep = CELL_SHEEP_1
        wolf = CELL_WOLF_2
        game_features = []
        # get positions of sheep, wolf and food items
        food = []
        x = 0
        sheep_position = [0, 0]
        for field_row in field:
            y = 0
            for item in field_row:
                if item == sheep:
                    sheep_position = [x, y]
                elif item == wolf:
                    wolf_position = (x, y)
                elif item == CELL_RHUBARB or item == CELL_GRASS:
                    food.append((x, y))
                y += 1
            x += 1
        dist = abs(sheep_position[1] - wolf_position[1]) + abs(sheep_position[0] - wolf_position[0])
        s_feature0 = float(dist)
        game_features.append(s_feature0)

        # feature 1: determine if wolf within two steps up
        if sheep_position[1] - wolf_position[1] < 0:
            s_feature1 = 1.0
        else:
            s_feature1 = 0.0
        game_features.append(s_feature1)

        # feature 2: determine if wolf within two steps down
        if sheep_position[1] - wolf_position[1] > 0:
            s_feature2 = 1.0
        else:
            s_feature2 = 0.0
        game_features.append(s_feature2)

        # feature 3: determine if wolf within two steps left
        if sheep_position[0] - wolf_position[0] < 0:
            s_feature3 = 1.0
        else:
            s_feature3 = 0.0
        game_features.append(s_feature3)

        # feature 4: determine if wolf within two steps right
        if sheep_position[0] - wolf_position[0] > 0:
            s_feature4 = 1.0
        else:
            s_feature4 = 0.0
        game_features.append(s_feature4)

        # feature 5-8: moves possible
        if sheep_position[0] != 0 and field[sheep_position[0] - 1][sheep_position[1]] in [CELL_GRASS, CELL_EMPTY,
                                                                                          CELL_RHUBARB]:
            s_feature5 = 1.0
        else:
            s_feature5 = 0.0
        game_features.append(s_feature5)

        if sheep_position[0] != FIELD_HEIGHT - 1 and field[sheep_position[0] + 1][sheep_position[1]] in [CELL_GRASS,
                                                                                                         CELL_EMPTY,
                                                                                                         CELL_RHUBARB]:
            s_feature6 = 1.0
        else:
            s_feature6 = 0.0
        game_features.append(s_feature6)

        if sheep_position[1] != 0 and field[sheep_position[0]][sheep_position[1] - 1] in [CELL_GRASS, CELL_EMPTY,
                                                                                          CELL_RHUBARB]:
            s_feature7 = 1.0
        else:
            s_feature7 = 0.0
        game_features.append(s_feature7)

        if sheep_position[1] != FIELD_WIDTH - 1 and field[sheep_position[0]][sheep_position[1] + 1] in [CELL_GRASS,
                                                                                                        CELL_EMPTY,
                                                                                                        CELL_RHUBARB]:
            s_feature8 = 1.0
        else:
            s_feature8 = 0.0
        game_features.append(s_feature8)

        s_feature9 = 0.0
        s_feature10 = 0.0
        s_feature11 = 0.0
        s_feature12 = 0.0
        # determine closest food:
        food_distance = 1000
        food_goal = None
        for food_item in food:
            distance = abs(food_item[0] - sheep_position[0]) + abs(food_item[1] - sheep_position[1])
            if distance < food_distance:
                food_distance = distance
                food_goal = food_item

        if food_goal != None:
            res = KingsKeep.astar(field, sheep_position, food_goal, ['.', 'g', 'r'])
            if res != None:

                res = res[::-1]
                direction = KingsKeep.getDirection(sheep_position, res[0])

                if direction == 1:
                    s_feature9 = 1.0

                if direction == -1:
                    s_feature10 = 1.0

                if direction == 2:
                    s_feature11 = 1.0

                if direction == -2:
                    s_feature12 = 1.0

        game_features.append(s_feature9)
        game_features.append(s_feature10)
        game_features.append(s_feature11)
        game_features.append(s_feature12)

        s_feature13 = 0.0
        if food_goal and field[food_goal[0]][food_goal[1]] == CELL_RHUBARB:
            s_feature13 = 1.0
        game_features.append(s_feature13)
        print(game_features)
        nested_state = [[FIELD_ELEMENTS_MAPPER[j] for j in i] for i in field]
        flat_state = [item for sublist in nested_state for item in sublist]
        game_features.extend(flat_state)
        return torch.tensor([game_features]).to(device)

    @staticmethod
    def convert_field_to_state_wolf(field, device):
        for x in field:
            print(x, sep=" ")
        sheep = CELL_SHEEP_2
        wolf = CELL_WOLF_1
        game_features = []
        # get positions of sheep, wolf and food items
        food = []
        x = 0
        sheep_position = [0, 0]
        for field_row in field:
            y = 0
            for item in field_row:
                if item == sheep:
                    sheep_position = [x, y]
                elif item == wolf:
                    wolf_position = (x, y)
                elif item == CELL_RHUBARB or item == CELL_GRASS:
                    food.append((x, y))
                y += 1
            x += 1
        dist = abs(sheep_position[1] - wolf_position[1]) + abs(sheep_position[0] - wolf_position[0])
        s_feature0 = float(dist)
        game_features.append(s_feature0)

        # feature 1: determine if wolf within two steps up
        if sheep_position[1] - wolf_position[1] < 0:
            s_feature1 = 1.0
        else:
            s_feature1 = 0.0
        game_features.append(s_feature1)

        # feature 2: determine if wolf within two steps down
        if sheep_position[1] - wolf_position[1] > 0:
            s_feature2 = 1.0
        else:
            s_feature2 = 0.0
        game_features.append(s_feature2)

        # feature 3: determine if wolf within two steps left
        if sheep_position[0] - wolf_position[0] < 0:
            s_feature3 = 1.0
        else:
            s_feature3 = 0.0
        game_features.append(s_feature3)

        # feature 4: determine if wolf within two steps right
        if sheep_position[0] - wolf_position[0] > 0:
            s_feature4 = 1.0
        else:
            s_feature4 = 0.0
        game_features.append(s_feature4)

        # feature 5-8: moves possible
        if sheep_position[0] != 0 and field[sheep_position[0] - 1][sheep_position[1]] in [CELL_GRASS, CELL_EMPTY,
                                                                                          CELL_RHUBARB]:
            s_feature5 = 1.0
        else:
            s_feature5 = 0.0
        game_features.append(s_feature5)

        if sheep_position[0] != FIELD_HEIGHT - 1 and field[sheep_position[0] + 1][sheep_position[1]] in [CELL_GRASS,
                                                                                                         CELL_EMPTY,
                                                                                                         CELL_RHUBARB]:
            s_feature6 = 1.0
        else:
            s_feature6 = 0.0
        game_features.append(s_feature6)

        if sheep_position[1] != 0 and field[sheep_position[0]][sheep_position[1] - 1] in [CELL_GRASS, CELL_EMPTY,
                                                                                          CELL_RHUBARB]:
            s_feature7 = 1.0
        else:
            s_feature7 = 0.0
        game_features.append(s_feature7)

        if sheep_position[1] != FIELD_WIDTH - 1 and field[sheep_position[0]][sheep_position[1] + 1] in [CELL_GRASS,
                                                                                                        CELL_EMPTY,
                                                                                                        CELL_RHUBARB]:
            s_feature8 = 1.0
        else:
            s_feature8 = 0.0
        game_features.append(s_feature8)

        direction = 10
        if sheep_position[0] == 0 and sheep_position[1] == 0:
            res = KingsKeep.astar(field, wolf_position, sheep_position, ['.', 'g', 'r', CELL_SHEEP_2])
            if res is not None:
                res = res[::-1]
                direction = KingsKeep.getDirection(sheep_position, res[0])

        s_feature9 = 0.0
        s_feature10 = 0.0
        s_feature11 = 0.0
        s_feature12 = 0.0

        if direction == 1:
            s_feature9 = 1.0

        if direction == -1:
            s_feature10 = 1.0

        if direction == 2:
            s_feature11 = 1.0

        if direction == -2:
            s_feature12 = 1.0

        game_features.append(s_feature9)
        game_features.append(s_feature10)
        game_features.append(s_feature11)
        game_features.append(s_feature12)
        game_features.append(0)
        nested_state = [[FIELD_ELEMENTS_MAPPER[j] for j in i] for i in field]
        flat_state = [item for sublist in nested_state for item in sublist]
        game_features.extend(flat_state)
        return torch.tensor([game_features]).to(device)

    @staticmethod
    def convert_field_to_state(field, device, figure):
        print(figure)
        if figure == "sheep2" or figure == "wolf2":
            for line in field:
                if CELL_SHEEP_2 in line:
                    index1 = line.index(CELL_SHEEP_2)
                    line[index1] = "sheep1"
                if CELL_SHEEP_1 in line:
                    index2 = line.index(CELL_SHEEP_1)
                    line[index2] = CELL_SHEEP_2

                if "sheep1" in line:
                    line[index1] = CELL_SHEEP_1

                if CELL_WOLF_2 in line:
                    index1 = line.index(CELL_WOLF_2)
                    line[index1] = "wolf1"
                if CELL_WOLF_1 in line:
                    index2 = line.index(CELL_WOLF_1)
                    line[index2] = CELL_WOLF_2

                if "wolf1" in line:
                    line[index1] = CELL_WOLF_1

        if figure == "sheep1" or figure == "sheep2":
            return KingsKeep.convert_field_to_state_sheep(field, device)
        else:
            return KingsKeep.convert_field_to_state_wolf(field, device)

    def get_device(self):
        return torch.device("cpu")

    def get_sheep_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('rlplayer_sheep_model.pt')
        model = DQN(n_inputs=299)
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def get_wolf_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('rlplayer_wolf_model.pt')
        model = DQN(n_inputs=299)
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def move_sheep(self, p_num, p_state, p_time_remaining, field):
        if 'sheep_model' not in p_state:
            p_state['sheep_model'] = self.get_sheep_model()

        if 'device' not in p_state:
            p_state['device'] = self.get_device()
        if p_num == 1:
            sheep = "sheep1"
        else:
            sheep = "sheep2"
        state = self.convert_field_to_state(field=field, device=p_state['device'], figure=sheep)
        with torch.no_grad():
            move = p_state['sheep_model'](state).argmax(dim=1).to(p_state['device'])
        return int(move) - 2, p_state

    def move_wolf(self, p_num, p_state, p_time_remaining, field):
        if 'wolf_model' not in p_state:
            p_state['wolf_model'] = self.get_wolf_model()

        if 'device' not in p_state:
            p_state['device'] = self.get_device()


        if p_num == 1:
            wolf = "wolf1"
        else:
            wolf = "wolf2"
        state = self.convert_field_to_state(field=field, device=p_state['device'], figure=wolf)
        with torch.no_grad():
            move = p_state['wolf_model'](state).argmax(dim=1).to(p_state['device'])
        return int(move) - 2, p_state

    @staticmethod
    def astar(array, start, goal, valid):
        startPosi = (start[0], start[1])
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), ]

        close_set = set()

        came_from = {}

        gscore = {startPosi: 0}

        fscore = {startPosi: ((goal[0] - startPosi[0]) ** 2 + (goal[1] - startPosi[1]) ** 2) ** 0.5}

        oheap = []

        oheap.insert(0, (fscore[startPosi], startPosi))

        while oheap:
            oheap.sort(key=lambda dist: dist[0])
            current = oheap.pop(0)[1]

            if current[0] == goal[0] and current[1] == goal[1]:

                data = []

                while current in came_from:
                    data.append(current)

                    current = came_from[current]

                return data

            close_set.add(current)

            for i, j in neighbors:

                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + 1

                if not (0 <= neighbor[0] < FIELD_HEIGHT) or not (0 <= neighbor[1] < FIELD_WIDTH) or array[neighbor[0]][neighbor[1]] not in valid:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore[neighbor]:
                    continue

                if tentative_g_score < gscore[current] or neighbor not in [i[1] for i in oheap]:
                    if array[neighbor[0]][neighbor[1]] == CELL_RHUBARB:
                        tentative_g_score -= 0.75
                    came_from[neighbor] = current

                    gscore[neighbor] = tentative_g_score

                    fscore[neighbor] = tentative_g_score + ((goal[0] - startPosi[0]) ** 2 + (goal[1] - startPosi[1]) ** 2) ** 0.5
                    oheap.insert(0, (fscore[neighbor], neighbor))
    @staticmethod
    def getDirection(figure_position, target):
        delta_x = figure_position[0] - target[0]
        delta_y = figure_position[1] - target[1]
        if delta_x == 0 and delta_y == 1:
            return MOVE_LEFT
        if delta_x == 0 and delta_y == -1:
            return MOVE_RIGHT
        if delta_x == 1 and delta_y == 0:
            return MOVE_UP
        if delta_x == -1 and delta_y == 0:
            return MOVE_DOWN