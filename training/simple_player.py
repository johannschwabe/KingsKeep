from config import *
import pickle
import pathlib


def get_class_name():
    return 'SimplePlayer'


class SimplePlayer:
    def __init__(self):
        self.name = "Simple Player"
        self.uzh_shortname = "splayer"
        self.sheep_model = None
        self.wolf_model = None

    def get_sheep_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('splayer_sheep_model.sav')
        return pickle.load(open(file_path, 'rb'))

    def get_wolf_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('splayer_wolf_model.sav')
        return pickle.load(open(file_path, 'rb'))

    def move_sheep(self, p_num, field):
        if not self.sheep_model:
            self.sheep_model = self.get_sheep_model()

        sheep_model = self.sheep_model

        X_sheep = []

        # preprocess field to get features, add to X_field
        # this code is largely copied from the Jupyter Notebook where the models were trained

        # create empty feature array for this game state
        game_features = []

        if p_num == 1:
            sheep = CELL_SHEEP_1
            wolf = CELL_WOLF_2
        else:
            sheep = CELL_SHEEP_2
            wolf = CELL_WOLF_1

        # get positions of sheep, wolf and food items
        food = []
        y = 0
        sheep_position = [0, 0]
        wolf_position = [0, 0]
        for field_row in field:
            x = 0
            for item in field_row:
                if item == sheep:
                    sheep_position = (x, y)
                elif item == wolf:
                    wolf_position = (x, y)
                elif item == CELL_RHUBARB or item == CELL_GRASS:
                    food.append((x, y))
                x += 1
            y += 1

        # feature 1: determine if wolf within two steps up
        if sheep_position[1] - wolf_position[1] <= 2 and sheep_position[1] - wolf_position[1] > 0:
            s_feature1 = 1
        else:
            s_feature1 = 0
        game_features.append(s_feature1)

        # feature 2: determine if wolf within two steps down
        if sheep_position[1] - wolf_position[1] >= -2 and sheep_position[1] - wolf_position[1] < 0:
            s_feature2 = 1
        else:
            s_feature2 = 0
        game_features.append(s_feature2)

        # feature 3: determine if wolf within two steps left
        if sheep_position[0] - wolf_position[0] <= 2 and sheep_position[0] - wolf_position[0] > 0:
            s_feature3 = 1
        else:
            s_feature3 = 0
        game_features.append(s_feature3)

        # feature 4: determine if wolf within two steps right
        if sheep_position[0] - wolf_position[0] >= -2 and sheep_position[0] - wolf_position[0] < 0:
            s_feature4 = 1
        else:
            s_feature4 = 0
        game_features.append(s_feature4)

        s_feature5 = 0
        s_feature6 = 0
        s_feature7 = 0
        s_feature8 = 0

        # determine closest food:
        food_distance = 1000
        food_goal = None

        for food_item in food:
            distance = abs(food_item[0] - sheep_position[0]) + abs(food_item[1] - sheep_position[1])
            if distance < food_distance:
                food_distance = distance
                food_goal = food_item

        if food_goal != None:
            # feature 5: determine if closest food is below the sheep
            if sheep_position[1] - food_goal[1] < 0:
                s_feature5 = 1

            # feature 6: determine if closest food is above the sheep
            if sheep_position[1] - food_goal[1] > 0:
                s_feature6 = 1

            # feature 7: determine if closest food is right of the sheep
            if sheep_position[0] - food_goal[0] < 0:
                s_feature7 = 1

            # feature 8: determine if closest food is left of the sheep
            if sheep_position[0] - food_goal[0] > 0:
                s_feature8 = 1

        game_features.append(s_feature5)
        game_features.append(s_feature6)
        game_features.append(s_feature7)
        game_features.append(s_feature8)

        # add features and move to X_sheep and Y_sheep
        X_sheep.append(game_features)

        result = sheep_model.predict(X_sheep)

        return int(result)

    def move_wolf(self, p_num, field):
        if p_num == 1:
            sheep = CELL_SHEEP_2
            wolf = CELL_WOLF_1
        else:
            sheep = CELL_SHEEP_1
            wolf = CELL_WOLF_2

        # get positions of sheep, wolf and food items
        x = 0
        sheep_position = [0, 0]
        wolf_position = [0, 0]
        for field_row in field:
            y = 0
            for item in field_row:
                if item == sheep:
                    sheep_position = (x, y)
                elif item == wolf:
                    wolf_position = (x, y)
                y += 1
            x += 1

        res = SimplePlayer.astar(field, wolf_position, sheep_position,
                                 [CELL_GRASS, CELL_EMPTY, CELL_RHUBARB, sheep])
        if res != None and res != []:
            res = res[::-1]
            return SimplePlayer.getDirection(wolf_position, res[0])
        return 0

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