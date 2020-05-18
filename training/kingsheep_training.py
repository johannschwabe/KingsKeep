import copy
from config import *


class KingsheepEnv:
    def __init__(self, player1, player2, map_name):
        self.player1 = player1
        self.player2 = player2
        self.ks = KsField(map_name, name1=self.player1.name, name2=self.player2.name)
        self.iteration = 0

    def step(self):
        self.iteration += 1
        iteration_summary, game_over = _kingsheep_iteration(i=self.iteration,
                                                            ks=self.ks,
                                                            player1=self.player1,
                                                            player2=self.player2)
        if self.iteration >= N_ITERATIONS:
            game_over = True

        return iteration_summary, game_over


class KsField:
    def __init__(self, filepath, name1='Player 1', name2='Player 2'):
        # initialize the field with empty cells.
        self.field = [[CELL_EMPTY for x in range(FIELD_WIDTH)] for y in range(FIELD_HEIGHT)]
        self.read_field(filepath)
        self.score1 = 0
        self.score2 = 0
        self.grading1 = 0
        self.grading2 = 0
        self.name1 = name1
        self.name2 = name2

    # Field related functions
    def read_field(self, fp):
        file = open(fp, 'r')
        for lineno, line in enumerate(file, 1):
            # turn the line into a string, strip the tangling \n and then assign it to the field variable
            self.field[lineno - 1] = list(str(line).strip('\n'))

    def get_field(self):
        return copy.deepcopy(self.field)

    def get_position(self, figure):
        # Next statement is a list comprehension.
        # it first generated a list of all x's in self. field, where figure is in that particular x
        # it then returns the first element (which is the row that contaions the figure),
        # as we know that there is only one of each of these figures
        x = [x for x in self.field if figure in x][0]
        return (self.field.index(x), x.index(figure))

    def new_position(self, x_old, y_old, move):
        if move == MOVE_LEFT:
            return (x_old, y_old - 1)
        elif move == MOVE_RIGHT:
            return (x_old, y_old + 1)
        elif move == MOVE_UP:
            return (x_old - 1, y_old)
        elif move == MOVE_DOWN:
            return (x_old + 1, y_old)

    def valid(self, figure, x_new, y_new):
        if x_new > FIELD_HEIGHT - 1:
            return False
        elif x_new < 0:
            return False
        elif y_new > FIELD_WIDTH - 1:
            return False
        elif y_new < 0:
            return False

        # Neither the sheep nor the wolf, can enter a square with a fence on.
        if self.field[x_new][y_new] == CELL_FENCE:
            return False

        # Wolfs can not step on squares occupied by the opponents wolf (wolfs block each other).
        # Wolfs can not step on squares occupied by the sheep of the same player .
        if figure == CELL_WOLF_1:
            if self.field[x_new][y_new] == CELL_WOLF_2:
                return False
            elif self.field[x_new][y_new] == CELL_SHEEP_1:
                return False
        elif figure == CELL_WOLF_2:
            if self.field[x_new][y_new] == CELL_WOLF_1:
                return False
            elif self.field[x_new][y_new] == CELL_SHEEP_2:
                return False

        # Sheep can not step on squares occupied by the wolf of the same player.
        # Sheep can not step on squares occupied by the opposite sheep.
        if figure == CELL_SHEEP_1:
            if self.field[x_new][y_new] == CELL_SHEEP_2 or \
                    self.field[x_new][y_new] == CELL_WOLF_1:
                return False
        elif figure == CELL_SHEEP_2:
            if self.field[x_new][y_new] == CELL_SHEEP_1 or \
                    self.field[x_new][y_new] == CELL_WOLF_2:
                return False

        return True

    def award(self, figure):
        if figure == CELL_RHUBARB:
            return AWARD_RHUBARB
        elif figure == CELL_GRASS:
            return AWARD_GRASS
        else:
            return 0

    def move(self, figure, move):
        if move != MOVE_NONE:
            (x_old, y_old) = self.get_position(figure)
            (x_new, y_new) = self.new_position(x_old, y_old, move)

            if self.valid(figure, x_new, y_new):
                target_figure = self.field[x_new][y_new]

                # wolf of player1 catches the sheep of player2 the game ends immediately and player1 wins and
                # is awarded all the points for the current run and vice versa

                # If the sheep steps on a food object, the food object is consumed (removed from the map) and a score
                # is awarded.

                if figure == CELL_SHEEP_1:
                    if target_figure == CELL_WOLF_2:
                        self.field[x_old][y_old] = CELL_SHEEP_1_d
                        self.score2 += self.score1
                        self.score1 = -1
                        return True
                    else:
                        self.score1 += self.award(target_figure)

                elif figure == CELL_SHEEP_2:
                    if target_figure == CELL_WOLF_1:
                        self.field[x_old][y_old] = CELL_SHEEP_2_d
                        self.score1 += self.score2
                        self.score2 = -1
                        return True
                    else:
                        self.score2 += self.award(target_figure)

                # If the wolf steps on a food object, the food object gets removed but no score is awarded.

                elif figure == CELL_WOLF_1:
                    if target_figure == CELL_SHEEP_2:
                        self.field[x_new][y_new] = CELL_SHEEP_2_d
                        self.score1 += self.score2
                        self.score2 = -1
                        return True

                elif figure == CELL_WOLF_2:
                    if target_figure == CELL_SHEEP_1:
                        self.field[x_new][y_new] = CELL_SHEEP_1_d
                        self.score2 += self.score1
                        self.score1 = -1
                        return True

                # actual figure move
                self.field[x_new][y_new] = figure
                self.field[x_old][y_old] = CELL_EMPTY
                return False

            else:  # if move is not valid
                return False

        else:  # if move = none
            return False


def _compute_move(f_move, ks, p_num, figure, game_over):
    move = f_move(p_num, ks.get_field())
    result_game_over = ks.move(figure, move)
    game_over = game_over or result_game_over

    return game_over, move


def _kingsheep_iteration(i, ks, player1, player2):
    game_over = False
    iteration_summary = {}

    # sheep1 move
    score_before = ks.score1
    iteration_summary['sheep1'] = {}
    iteration_summary['sheep1']['state'] = ks.field.copy()
    game_over, move = _compute_move(f_move=player1.move_sheep,
                                    ks=ks,
                                    p_num=1,
                                    figure=CELL_SHEEP_1,
                                    game_over=game_over)
    iteration_summary['sheep1']['move'] = int(move) + 2
    iteration_summary['sheep1']['reward'] = ks.score1 - score_before
    iteration_summary['sheep1']['next_state'] = ks.field.copy()

    if not game_over:
        # sheep2 move
        score_before = ks.score2
        iteration_summary['sheep2'] = {}
        iteration_summary['sheep2']['state'] = ks.field.copy()
        game_over, move = _compute_move(f_move=player2.move_sheep,
                                        ks=ks,
                                        p_num=2,
                                        figure=CELL_SHEEP_2,
                                        game_over=game_over)
        iteration_summary['sheep2']['move'] = int(move) + 2
        iteration_summary['sheep2']['reward'] = ks.score2 - score_before
        iteration_summary['sheep2']['next_state'] = ks.field.copy()

    if i % 2 == 0 and not game_over:
        # wolf1 move
        score_before = ks.score1
        iteration_summary['wolf1'] = {}
        iteration_summary['wolf1']['state'] = ks.field.copy()
        game_over, move = _compute_move(f_move=player1.move_wolf,
                                        ks=ks,
                                        p_num=1,
                                        figure=CELL_WOLF_1,
                                        game_over=game_over)
        iteration_summary['wolf1']['move'] = int(move) + 2
        iteration_summary['wolf1']['reward'] = ks.score1 - score_before
        iteration_summary['wolf1']['next_state'] = ks.field.copy()

    if i % 2 == 0 and not game_over:
        # wolf2 move
        score_before = ks.score2
        iteration_summary['wolf2'] = {}
        iteration_summary['wolf2']['state'] = ks.field.copy()
        game_over, move = _compute_move(f_move=player2.move_wolf,
                                        ks=ks,
                                        p_num=2,
                                        figure=CELL_WOLF_2,
                                        game_over=game_over)
        iteration_summary['wolf2']['move'] = int(move) + 2
        iteration_summary['wolf2']['reward'] = ks.score2 - score_before
        iteration_summary['wolf2']['next_state'] = ks.field.copy()

    return iteration_summary, game_over
