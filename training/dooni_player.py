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
    return 'DooniPlayer'


class DQN(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()

        self.fc1 = nn.Linear(in_features=n_inputs, out_features=14)
        self.fc2 = nn.Linear(in_features=14, out_features=8)
        self.out = nn.Linear(in_features=8, out_features=5)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class DooniPlayer:
    def __init__(self):
        self.name = "Dooni Player"
        self.uzh_shortname = "ddemet"

    @staticmethod
    def convert_field_to_state(field, p_num, device):
        # nested_state = [[FIELD_ELEMENTS_MAPPER[j] for j in i] for i in field]
        # flat_state = [item for sublist in nested_state for item in sublist]
        # return torch.tensor([flat_state]).to(device)
        state = [0.0] * 34

        s_feature1 = 0
        s_feature2 = 0
        s_feature3 = 0
        s_feature4 = 0
        s_feature5 = 0
        s_feature6 = 0
        s_feature7 = 0
        s_feature8 = 0
        s_feature9 = 0
        s_feature10 = 0
        s_feature11 = 0
        s_feature12 = 0
        s_feature13 = 0
        s_feature14 = 0
        s_feature15 = 0
        s_feature16 = 0
        s_feature17 = 0
        s_feature18 = 0
        s_feature19 = 0
        s_feature20 = 0
        s_feature21 = 0
        s_feature22 = 0
        s_feature23 = 0
        s_feature24 = 0
        s_feature25 = 0
        s_feature26 = 0
        s_feature27 = 0
        s_feature28 = 0
        s_feature33 = 0
        s_feature34 = 0

        rhubarb = CELL_RHUBARB
        grass = CELL_GRASS
        mySheepPos = (0, 0)
        myWolfPos = (0, 0)
        otherSheepPos = (0, 0)
        otherWolfPos = (0, 0)
        mySheep = None
        myWolf = None
        otherSheep = None
        otherWolf = None

        if p_num == 1:
            mySheep = CELL_SHEEP_1
            myWolf = CELL_WOLF_1
            otherSheep = CELL_SHEEP_2
            otherWolf = CELL_WOLF_2
        else:
            mySheep = CELL_SHEEP_2
            myWolf = CELL_WOLF_2
            otherSheep = CELL_SHEEP_1
            otherWolf = CELL_WOLF_1
        rhubarbs = []
        grasses = []
        fences = []
        y = 0
        for field_row in field:
            x = 0
            for item in field_row:
                if item == CELL_SHEEP_1:
                    s_feature17 = x / FIELD_WIDTH
                    s_feature18 = y / FIELD_HEIGHT
                elif item == CELL_SHEEP_2:
                    s_feature19 = x / FIELD_WIDTH
                    s_feature20 = y / FIELD_HEIGHT
                if item == CELL_WOLF_1:
                    s_feature21 = x / FIELD_WIDTH
                    s_feature22 = y / FIELD_HEIGHT
                elif item == CELL_WOLF_2:
                    s_feature23 = x / FIELD_WIDTH
                    s_feature24 = y / FIELD_HEIGHT

                if item == mySheep:
                    mySheepPos = (x, y)
                elif item == myWolf:
                    myWolfPos = (x, y)
                elif item == otherSheep:
                    otherSheepPos = (x, y)
                elif item == otherWolf:
                    otherWolfPos = (x, y)
                elif item == CELL_RHUBARB:
                    rhubarbs.append((x, y))
                elif item == CELL_GRASS:
                    grasses.append((x, y))
                elif item == CELL_FENCE:
                    fences.append((x, y))
                x += 1
            y += 1

        # feature 1: determine if wolf within two steps up
        if mySheepPos[1] - otherWolfPos[1] <= 2 and mySheepPos[1] - otherWolfPos[1] > 0:
            s_feature1 = 1

        # feature 2: determine if wolf within two steps down
        if mySheepPos[1] - otherWolfPos[1] >= -2 and mySheepPos[1] - otherWolfPos[1] < 0:
            s_feature2 = 1

        # feature 3: determine if wolf within two steps left
        if mySheepPos[0] - otherWolfPos[0] <= 2 and mySheepPos[0] - otherWolfPos[0] > 0:
            s_feature3 = 1

        # feature 4: determine if wolf within two steps right
        if mySheepPos[0] - otherWolfPos[0] >= -2 and mySheepPos[0] - otherWolfPos[0] < 0:
            s_feature4 = 1

        # determine closest food:
        rhubarb_distance = 1000
        rhubarb_goal = None
        for rhu in rhubarbs:
            distance = abs(rhu[0] - mySheepPos[0]) + abs(rhu[1] - mySheepPos[1])
            if distance < rhubarb_distance:
                rhubarb_distance = distance
                rhubarb_goal = rhu

        if rhubarb_goal != None:
            # feature 5: determine if closest food is below the sheep
            if mySheepPos[1] - rhubarb_goal[1] < 0:
                s_feature5 = 1

            # feature 6: determine if closest food is above the sheep
            if mySheepPos[1] - rhubarb_goal[1] > 0:
                s_feature6 = 1

            # feature 7: determine if closest food is right of the sheep
            if mySheepPos[0] - rhubarb_goal[0] < 0:
                s_feature7 = 1

            # feature 8: determine if closest food is left of the sheep
            if mySheepPos[0] - rhubarb_goal[0] > 0:
                s_feature8 = 1

        # determine closest food:
        grass_distance = 1000
        grass_goal = None
        for gra in grasses:
            distance = abs(gra[0] - mySheepPos[0]) + abs(gra[1] - mySheepPos[1])
            if distance < grass_distance:
                grass_distance = distance
                grass_goal = gra

        if grass_goal != None:
            # feature 5: determine if closest food is below the sheep
            if mySheepPos[1] - grass_goal[1] < 0:
                s_feature9 = 1

            # feature 6: determine if closest food is above the sheep
            if mySheepPos[1] - grass_goal[1] > 0:
                s_feature10 = 1

            # feature 7: determine if closest food is right of the sheep
            if mySheepPos[0] - grass_goal[0] < 0:
                s_feature11 = 1

            # feature 8: determine if closest food is left of the sheep
            if mySheepPos[0] - grass_goal[0] > 0:
                s_feature12 = 1

        for fen in fences:
            if fen[0] == mySheepPos[0] and fen[1] - mySheepPos[1] == 1:
                s_feature13 = 1
            elif fen[0] == mySheepPos[0] and fen[1] - mySheepPos[1] == -1:
                s_feature14 = 1
            elif fen[1] == mySheepPos[1] and fen[0] - mySheepPos[0] == 1:
                s_feature15 = 1
            elif fen[1] == mySheepPos[1] and fen[0] - mySheepPos[0] == -1:
                s_feature16 = 1
            if fen[0] == myWolfPos[0] and fen[1] - myWolfPos[1] == 1:
                s_feature25 = 1
            elif fen[0] == myWolfPos[0] and fen[1] - myWolfPos[1] == -1:
                s_feature26 = 1
            elif fen[1] == myWolfPos[1] and fen[0] - myWolfPos[0] == 1:
                s_feature27 = 1
            elif fen[1] == myWolfPos[1] and fen[0] - myWolfPos[0] == -1:
                s_feature28 = 1
        if mySheepPos[0] == myWolfPos[0] and mySheepPos[1] - myWolfPos[1] == 1:
            s_feature25 = 1
        elif mySheepPos[0] == myWolfPos[0] and mySheepPos[1] - myWolfPos[1] == -1:
            s_feature26 = 1
        elif mySheepPos[1] == myWolfPos[1] and mySheepPos[0] - myWolfPos[0] == 1:
            s_feature27 = 1
        elif mySheepPos[1] == myWolfPos[1] and mySheepPos[0] - myWolfPos[0] == -1:
            s_feature28 = 1

        if myWolfPos[0] == mySheepPos[0] and myWolfPos[1] - mySheepPos[1] == 1:
            s_feature13 = 1
        elif myWolfPos[0] == mySheepPos[0] and myWolfPos[1] - mySheepPos[1] == -1:
            s_feature14 = 1
        elif myWolfPos[1] == mySheepPos[1] and myWolfPos[0] - mySheepPos[0] == 1:
            s_feature15 = 1
        elif myWolfPos[1] == mySheepPos[1] and myWolfPos[0] - mySheepPos[0] == -1:
            s_feature16 = 1
        if otherSheepPos[0] == mySheepPos[0] and otherSheepPos[1] - mySheepPos[1] == 1:
            s_feature13 = 1
        elif otherSheepPos[0] == mySheepPos[0] and otherSheepPos[1] - mySheepPos[1] == -1:
            s_feature14 = 1
        elif otherSheepPos[1] == mySheepPos[1] and otherSheepPos[0] - mySheepPos[0] == 1:
            s_feature15 = 1
        elif otherSheepPos[1] == mySheepPos[1] and otherSheepPos[0] - mySheepPos[0] == -1:
            s_feature16 = 1

        if otherSheepPos[0] > myWolfPos[0]:
            s_feature33 = 1
        if otherSheepPos[1] < myWolfPos[1]:
            s_feature34 = 1

        upPoints = 0
        rightPoints = 0
        downPoints = 0
        leftPoints = 0
        for gra in grasses:
            if mySheepPos[1] - gra[1] < 0:
                downPoints += 1

            if mySheepPos[1] - gra[1] > 0:
                upPoints += 1

            if mySheepPos[0] - gra[0] < 0:
                rightPoints += 1

            if mySheepPos[0] - gra[0] > 0:
                leftPoints += 1
        for rhu in rhubarbs:
            if mySheepPos[1] - rhu[1] < 0:
                downPoints += 5

            if mySheepPos[1] - rhu[1] > 0:
                upPoints += 5

            if mySheepPos[0] - rhu[0] < 0:
                rightPoints += 5

            if mySheepPos[0] - rhu[0] > 0:
                leftPoints += 5
        maxPoints = max(upPoints, rightPoints, downPoints, leftPoints, 1)
        upPoints /= maxPoints
        rightPoints /= maxPoints
        downPoints /= maxPoints
        leftPoints /= maxPoints

        state[0] = float(s_feature1)
        state[1] = float(s_feature2)
        state[2] = float(s_feature3)
        state[3] = float(s_feature4)
        state[4] = float(s_feature5)
        state[5] = float(s_feature6)
        state[6] = float(s_feature7)
        state[7] = float(s_feature8)
        state[8] = float(s_feature9)
        state[9] = float(s_feature10)
        state[10] = float(s_feature11)
        state[11] = float(s_feature12)
        state[12] = float(s_feature13)
        state[13] = float(s_feature14)
        state[14] = float(s_feature15)
        state[15] = float(s_feature16)
        state[16] = float(s_feature17)
        state[17] = float(s_feature18)
        state[18] = float(s_feature19)
        state[19] = float(s_feature20)
        state[20] = float(s_feature21)
        state[21] = float(s_feature22)
        state[22] = float(s_feature23)
        state[23] = float(s_feature24)
        state[24] = float(s_feature25)
        state[25] = float(s_feature26)
        state[26] = float(s_feature27)
        state[27] = float(s_feature28)
        state[28] = float(upPoints)
        state[29] = float(rightPoints)
        state[30] = float(downPoints)
        state[31] = float(leftPoints)
        state[32] = float(s_feature33)
        state[33] = float(s_feature34)
        # nested_state = [[FIELD_ELEMENTS_MAPPER[j] for j in i] for i in field]
        # flat_state = [item for sublist in nested_state for item in sublist]
        # print(flat_state)
        # flat_state.extend(state)
        # print("from script:", len(flat_state))
        return torch.tensor([state]).to(device)

    def get_device(self):
        return torch.device("cpu")

    def get_sheep_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('dooniplayer_sheep_model3.pt')
        model = DQN(n_inputs=34)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model

    def get_wolf_model(self):
        file_path = pathlib.Path(__file__).parent.absolute().joinpath('dooniplayer_wolf_model3.pt')
        model = DQN(n_inputs=34)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model

    def move_sheep(self, p_num, p_state, p_time_remaining, field):
        if 'sheep_model' not in p_state:
            p_state['sheep_model'] = self.get_sheep_model()

        if 'device' not in p_state:
            p_state['device'] = self.get_device()

        state = self.convert_field_to_state(field=field, p_num=p_num, device=p_state['device'])
        with torch.no_grad():
            move = p_state['sheep_model'](state).argmax(dim=1).to(p_state['device'])
        return int(move) - 2, p_state

    def move_wolf(self, p_num, p_state, p_time_remaining, field):
        if 'wolf_model' not in p_state:
            p_state['wolf_model'] = self.get_wolf_model()

        if 'device' not in p_state:
            p_state['device'] = self.get_device()

        state = self.convert_field_to_state(field=field, p_num=p_num, device=p_state['device'])
        with torch.no_grad():
            move = p_state['wolf_model'](state).argmax(dim=1).to(p_state['device'])
        return int(move) - 2, p_state
