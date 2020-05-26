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


class KingsKeep:
    def __init__(self, sheep_policy_net, wolf_policy_net, strategy, device):
        self.name = "RL Player"
        self.sheep_policy_net = sheep_policy_net
        self.wolf_policy_net = wolf_policy_net
        self.current_step = 0
        self.strategy = strategy
        self.device = device

    @staticmethod
    def convert_field_to_state_sheep(field, device):
        sheep = CELL_SHEEP_1
        wolf = CELL_WOLF_2
        game_features = []
        # get positions of sheep, wolf and food items
        food = []
        x = 0
        sheep_position = [0, 0]
        wolf_position = [0, 0]
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
            if res != None and res != []:

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

        subField = []

        for x in range(7):
            for y in range(9):
                count = 0
                x_field = 1+2*x
                y_field = 1+2*y
                count += KingsKeep.award(field[x_field - 1][y_field - 1])
                count += KingsKeep.award(field[x_field][y_field - 1])
                count += KingsKeep.award(field[x_field + 1][y_field - 1])
                count += KingsKeep.award(field[x_field - 1][y_field])
                count += KingsKeep.award(field[x_field][y_field])
                count += KingsKeep.award(field[x_field + 1][y_field])
                count += KingsKeep.award(field[x_field - 1][y_field + 1])
                count += KingsKeep.award(field[x_field][y_field + 1])
                count += KingsKeep.award(field[x_field + 1][y_field + 1])
                subField.append(count)

        dir_x = float(sheep_position[0] - wolf_position[0])
        dir_y = float(sheep_position[1] - wolf_position[1])
        game_features.append(dir_x)
        game_features.append(dir_y)
        game_features.extend(subField)

        pos_x = sheep_position[0]
        pos_y = sheep_position[1]
        game_features.append(pos_x)
        game_features.append(pos_y)

        return torch.tensor([game_features]).to(device)


    @staticmethod
    def convert_field_to_state_wolf(field, device):

        sheep = CELL_SHEEP_2
        wolf = CELL_WOLF_1
        game_features = []
        # get positions of sheep, wolf and food items
        food = []
        x = 0
        sheep_position = [0, 0]
        wolf_position = [0, 0]
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
        if wolf_position[0] != 0 and field[wolf_position[0] - 1][wolf_position[1]] in [CELL_GRASS, CELL_EMPTY,
                                                                                          CELL_RHUBARB]:
            s_feature5 = 1.0
        else:
            s_feature5 = 0.0
        game_features.append(s_feature5)

        if wolf_position[0] != FIELD_HEIGHT - 1 and field[wolf_position[0] + 1][wolf_position[1]] in [CELL_GRASS,
                                                                                                         CELL_EMPTY,
                                                                                                         CELL_RHUBARB]:
            s_feature6 = 1.0
        else:
            s_feature6 = 0.0
        game_features.append(s_feature6)

        if wolf_position[1] != 0 and field[wolf_position[0]][wolf_position[1] - 1] in [CELL_GRASS, CELL_EMPTY,
                                                                                          CELL_RHUBARB]:
            s_feature7 = 1.0
        else:
            s_feature7 = 0.0
        game_features.append(s_feature7)

        if wolf_position[1] != FIELD_WIDTH - 1 and field[wolf_position[0]][wolf_position[1] + 1] in [CELL_GRASS,
                                                                                                        CELL_EMPTY,
                                                                                                        CELL_RHUBARB]:
            s_feature8 = 1.0
        else:
            s_feature8 = 0.0
        game_features.append(s_feature8)

        direction = 10
        if sheep_position[0] != 0 and sheep_position[1] != 0:
            res = KingsKeep.astar(field, wolf_position, sheep_position, ['.', 'g', 'r', CELL_SHEEP_2])
            if res is not None and res != []:
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
        return torch.tensor([game_features]).to(device)



    @staticmethod
    def convert_field_to_state(field, device, figure):
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
    @staticmethod
    def convert_field_to_state2(states, device, figure, batch_size):

        processed_states = []
        for x in range(batch_size):
            print(x)
            print(states[x])
            processed_states.append(KingsKeep.convert_field_to_state(states[x], device, figure))
        return torch.tensor([processed_states]).to(device)


    def compute_move(self, field, is_sheep_move, p_num):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            # explore
            action = random.randrange(5)
            move = torch.tensor([action]).to(self.device)
        else:
            # exploit
            if is_sheep_move:

                state = self.convert_field_to_state(field=field, device=self.device, figure="sheep"+ str(p_num))
            else:
                state = self.convert_field_to_state(field=field, device=self.device, figure="wolf"+ str(p_num))

            policy_net = self.sheep_policy_net if is_sheep_move else self.wolf_policy_net
            with torch.no_grad():
                move = policy_net(state).argmax(dim=1).to(self.device)

        return int(move) - 2

    def move_sheep(self, p_num, field):
        return self.compute_move(field=field, is_sheep_move=True, p_num=p_num)

    def move_wolf(self, p_num, field):
        return self.compute_move(field=field, is_sheep_move=False, p_num=p_num)

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

    @staticmethod
    def award(figure):
        if figure == CELL_RHUBARB:
            return AWARD_RHUBARB
        elif figure == CELL_GRASS:
            return AWARD_GRASS
        else:
            return 0