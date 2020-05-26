from config import *
import pickle
import pathlib


def get_class_name():
    return 'SimplePlayer'


class SimplePlayer:
    def __init__(self):
        self.name = "greedy Player"
        self.uzh_shortname = "splayer"
        self.sheep_model = None
        self.wolf_model = None



    def move_sheep(self, p_num, field):
        if p_num == 1:
            sheep = CELL_SHEEP_1
            wolf = CELL_WOLF_2
        else:
            sheep = CELL_SHEEP_2
            wolf = CELL_WOLF_1

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

        res = SimplePlayer.astar(field,sheep_position, SimplePlayer.closest_goal(sheep_position,field),[CELL_GRASS, CELL_EMPTY, CELL_RHUBARB])
        if res != None and res!=[]:
            res = res[::-1]
            return SimplePlayer.getDirection(sheep_position,res[0])
        return 0


    def move_wolf(self, p_num, field):
        if p_num == 1:
            sheep = CELL_SHEEP_2
            wolf = CELL_WOLF_1
        else:
            sheep = CELL_SHEEP_1
            wolf = CELL_WOLF_2

        # get positions of sheep, wolf and food items
        x = 0
        sheep_position=[0, 0]
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


        res = SimplePlayer.astar(field, wolf_position,sheep_position,
                                 [CELL_GRASS, CELL_EMPTY, CELL_RHUBARB, sheep])
        if res != None and res!=[]:
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

    @staticmethod
    def closest_goal(sheep_position, field):
        possible_goals = []



        # make list of possible goals

        y_position = 0
        for line in field:
            x_position = 0
            for item in line:
                if item == CELL_RHUBARB or item == CELL_GRASS:
                    possible_goals.append((y_position, x_position))
                x_position += 1
            y_position += 1

        # determine closest item and return
        distance = 1000
        final_goal = [0,0]
        for possible_goal in possible_goals:
            if (abs(possible_goal[0] - sheep_position[0]) + abs(possible_goal[1] - sheep_position[1])) < distance:
                distance = abs(possible_goal[0] - sheep_position[0]) + abs(possible_goal[1] - sheep_position[1])
                final_goal = (possible_goal)

        return final_goal