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
        if not self.wolf_model:
            self.wolf_model = self.get_wolf_model()

        wolf_model = self.wolf_model

        # create empty feature array for this game state
        game_features = []
        X_wolf = []

        if p_num == 1:
            sheep = CELL_SHEEP_2
            wolf = CELL_WOLF_1
        else:
            sheep = CELL_SHEEP_1
            wolf = CELL_WOLF_2

        # get positions of sheep, wolf and food items
        y = 0
        for field_row in field:
            x = 0
            for item in field_row:
                if item == sheep:
                    sheep_position = (x, y)
                elif item == wolf:
                    wolf_position = (x, y)
                x += 1
            y += 1

        # feature 1: determine if the sheep is above the wolf
        if wolf_position[1] - sheep_position[1] > 0:
            w_feature1 = 1
        else:
            w_feature1 = 0
        game_features.append(w_feature1)

        # feature 2: determine if the sheep is below the wolf
        if wolf_position[1] - sheep_position[1] < 0:
            w_feature2 = 1
        else:
            w_feature2 = 0
        game_features.append(w_feature2)

        # feature 3: determine if the sheep is left of the wolf
        if wolf_position[0] - sheep_position[0] > 0:
            w_feature3 = 1
        else:
            w_feature3 = 0
        game_features.append(w_feature3)

        # feature 4: determine if the sheep is right of the wolf
        if wolf_position[0] - sheep_position[0] < 0:
            w_feature4 = 1
        else:
            w_feature4 = 0
        game_features.append(w_feature4)

        # add features and move to X_wolf and Y_wolf
        X_wolf.append(game_features)

        result = wolf_model.predict(X_wolf)

        return int(result)
