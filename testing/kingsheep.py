import copy
import argparse
import importlib
import time
import os.path
from config import *
import arcade
from multiprocessing import Pool, TimeoutError


class KingsheepWindow(arcade.Window):
    """Graphical Interface for Kingsheep"""

    def __init__(self, name, iterations, ks, player1, player2, pool, reason):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, name)

        # Set the working directory (where we expect to find files) to the same
        # directory this .py file is in. You can leave this out of your own
        # code, but it is needed to easily run the examples using "python -m"
        # as mentioned at the top of this program.
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        self.iteration = 0
        self.iterations = iterations
        self.ks = ks
        self.player1 = player1
        self.player2 = player2
        self.field_width = FIELD_WIDTH
        self.field_height = FIELD_HEIGHT
        self.last_key = -1
        self.reason = reason

        self.sprites = None
        self.grass = []
        self.rhubarb = []
        self.fence = []

        self.pool = pool

        arcade.start_render()
        arcade.set_background_color(arcade.color.LIGHT_GREEN)
        arcade.finish_render()

        # If you have sprite lists, you should create them here,
        # and set them to None

    def quit(self):
        arcade.close_window()

    def make_player(self, player, path):
        sprite = arcade.Sprite(path, SPRITE_SCALING_PLAYER)
        sprite.center_x = 50
        sprite.center_y = 50
        return sprite

    def setup(self):
        # Create your sprites and sprite lists here

        self.sprites = {}
        self.sprites[CELL_SHEEP_1] = self.make_player(CELL_SHEEP_1, "resources/gfx/sheep1.png")
        self.sprites[CELL_SHEEP_2] = self.make_player(CELL_SHEEP_2, "resources/gfx/sheep2.png")
        self.sprites[CELL_WOLF_1] = self.make_player(CELL_WOLF_1, "resources/gfx/wolf1.png")
        self.sprites[CELL_WOLF_2] = self.make_player(CELL_WOLF_2, "resources/gfx/wolf2.png")

    def get_coordinates(self, x, y):
        screen_x = SCREEN_WIDTH / self.field_width * (y + 0.5)
        screen_y = SCREEN_HEIGHT - SCREEN_HEIGHT / self.field_height * (x + 0.5)
        return screen_x, screen_y

    def set_coordinates(self, sprite, x, y):
        x_n, y_n = self.get_coordinates(x, y)
        sprite.center_x = x_n
        sprite.center_y = y_n

    def on_draw(self):
        # This command should happen before we start drawing. It will clear
        # the screen to the background color, and erase what we drew last frame.
        arcade.start_render()

        field = self.ks.get_field()

        new_grass = []
        new_fence = []
        new_rhubarb = []

        for x in range(len(field)):
            for y in range(len(field[x])):
                fig = field[x][y]
                if fig == CELL_EMPTY:
                    pass
                elif fig == CELL_GRASS:
                    if len(self.grass) == 0:
                        s = self.make_player(CELL_GRASS, "resources/gfx/grass.png")
                    else:
                        s = self.grass.pop()
                    self.set_coordinates(s, x, y)
                    new_grass.append(s)

                elif fig == CELL_RHUBARB:
                    if len(self.rhubarb) == 0:
                        s = self.make_player(CELL_RHUBARB, "resources/gfx/rhubarb.png")
                    else:
                        s = self.rhubarb.pop()
                    self.set_coordinates(s, x, y)
                    new_rhubarb.append(s)

                elif fig == CELL_FENCE:
                    if len(self.fence) == 0:
                        s = self.make_player(CELL_FENCE, "resources/gfx/skigard.png")
                    else:
                        s = self.fence.pop()
                    self.set_coordinates(s, x, y)
                    new_fence.append(s)

                else:
                    if fig == CELL_SHEEP_1_d:
                        fig = CELL_SHEEP_1
                        print("Need to implement dead sheep")
                    if fig == CELL_SHEEP_2_d:
                        fig = CELL_SHEEP_2
                        print("Need to implement dead sheep")

                    self.set_coordinates(self.sprites[fig], x, y)

        self.grass = new_grass
        self.fence = new_fence
        self.rhubarb = new_rhubarb

        # Call draw() on all your sprite lists below
        [x.draw() for x in self.sprites.values()]
        [x.draw() for x in self.grass]
        [x.draw() for x in self.fence]
        [x.draw() for x in self.rhubarb]

        s = "Score:  " + self.ks.name1 + ": " + str(self.ks.score1) + "  " + \
            self.ks.name2 + ": " + str(self.ks.score2)
        arcade.draw_text(s, 10, 10, arcade.color.BLACK, 12)

        time.sleep(slowdown)

    def update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you'll call update() on the sprite lists that
        need it.
        """
        self.iteration += 1

        game_over, reason = _kingsheep_iteration(
            i=self.iteration, ks=self.ks, player1=self.player1, player2=self.player2, pool=self.pool,
            reason=self.reason)

        if debug:
            self.ks.print_ks()

        if self.iteration >= self.iterations or game_over or self.last_key == 113:
            print(reason)
            time.sleep(2 * slowdown)
            arcade.close_window()

    def on_key_press(self, key, key_modifiers):
        self.last_key = key

    def on_key_release(self, key, key_modifiers):
        pass

    def on_mouse_motion(self, x, y, delta_x, delta_y):
        pass

    def on_mouse_press(self, x, y, button, key_modifiers):
        pass

    def on_mouse_release(self, x, y, button, key_modifiers):
        pass


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
        self.p1_time_remaining = MAX_PLAYER_TIME
        self.p2_time_remaining = MAX_PLAYER_TIME
        self.p1_state = {}
        self.p2_state = {}

    # Field related functions

    def read_field(self, fp):
        file = open(fp, 'r')
        for lineno, line in enumerate(file, 1):
            # turn the line into a string, strip the tangling \n and then assign it to the field variable
            self.field[lineno - 1] = list(str(line).strip('\n'))

    def get_field(self):
        return copy.deepcopy(self.field)

    def print_ks(self):
        if verbosity > 3:
            i = -1
            for line in self.field:
                i = i + 1
                print('{:2d}  {}'.format(i, ''.join(line)))
            print('    0123456789012345678')

        if verbosity > 0:
            print('Scores: {}: {:3d}   {}: {:3d}'.format(self.name1, self.score1, self.name2, self.score2))

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

    def move(self, figure, move, reason):
        if figure in {CELL_SHEEP_1, CELL_WOLF_1} and self.p1_time_remaining < 0:
            return True, f'timeout1'

        if figure in {CELL_SHEEP_2, CELL_WOLF_2} and self.p2_time_remaining < 0:
            return True, f'timeout2'

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
                        return True, 'sheep1 suicide'
                    else:
                        self.score1 += self.award(target_figure)

                elif figure == CELL_SHEEP_2:
                    if target_figure == CELL_WOLF_1:
                        self.field[x_old][y_old] = CELL_SHEEP_2_d
                        self.score1 += self.score2
                        self.score2 = -1
                        return True, 'sheep2 suicide'
                    else:
                        self.score2 += self.award(target_figure)

                # If the wolf steps on a food object, the food object gets removed but no score is awarded.

                elif figure == CELL_WOLF_1:
                    if target_figure == CELL_SHEEP_2:
                        self.field[x_new][y_new] = CELL_SHEEP_2_d
                        self.score1 += self.score2
                        self.score2 = -1
                        return True, 'sheep2 eaten'

                elif figure == CELL_WOLF_2:
                    if target_figure == CELL_SHEEP_1:
                        self.field[x_new][y_new] = CELL_SHEEP_1_d
                        self.score2 += self.score1
                        self.score1 = -1
                        return True, 'sheep1 eaten'

                # actual figure move
                self.field[x_new][y_new] = figure
                self.field[x_old][y_old] = CELL_EMPTY
                return False, reason

            else:  # if move is not valid
                return False, reason

        else:  # if move = none
            return False, reason


def _compute_move_async(f_move, field, p_num, p_state, p_time_remaining, pool):
    r = pool.apply_async(f_move, (p_num, p_state, p_time_remaining, field))

    start_time = time.perf_counter()
    move, p_state = r.get(MAX_PLAYER_TIME * 2)
    time_taken = time.perf_counter() - start_time

    return move, time_taken, p_state


def _compute_move(f_move, ks, p_num, figure, game_over, reason, pool, i):
    try:
        if p_num == 1:
            move, time_taken, ks.p1_state = _compute_move_async(
                f_move=f_move, field=ks.get_field(), p_num=p_num, p_state=ks.p1_state,
                p_time_remaining=ks.p1_time_remaining, pool=pool)
            if i > 1:
                ks.p1_time_remaining -= time_taken
                if ks.p1_time_remaining < 0:
                    ks.score1 = -1
        else:
            move, time_taken, ks.p2_state = _compute_move_async(
                f_move=f_move, field=ks.get_field(), p_num=p_num, p_state=ks.p2_state,
                p_time_remaining=ks.p2_time_remaining, pool=pool)
            if i > 1:
                ks.p2_time_remaining -= time_taken
                if ks.p2_time_remaining < 0:
                    ks.score2 = -1

        result_game_over, result_reason = ks.move(figure, move, reason)
        if result_reason != '':
            reason = result_reason
        game_over = game_over or result_game_over
    except TimeoutError:
        if p_num == 1:
            reason = 'timeout1'
        else:
            reason = 'timeout2'

        game_over = True

    return game_over, reason


def _kingsheep_iteration(i, ks, player1, player2, pool, reason):
    game_over = False

    # sheep1 move
    game_over, reason = _compute_move(f_move=player1.move_sheep, ks=ks, p_num=1, figure=CELL_SHEEP_1,
                                      game_over=game_over, reason=reason, pool=pool, i=i)

    if not game_over:
        # sheep2 move
        game_over, reason = _compute_move(f_move=player2.move_sheep, ks=ks, p_num=2, figure=CELL_SHEEP_2,
                                          game_over=game_over, reason=reason, pool=pool, i=i)

    if i % 2 == 0 and not game_over:
        # wolf1 move
        game_over, reason = _compute_move(f_move=player1.move_wolf, ks=ks, p_num=1, figure=CELL_WOLF_1,
                                          game_over=game_over, reason=reason, pool=pool, i=i)

    if i % 2 == 0 and not game_over:
        # wolf2 move
        game_over, reason = _compute_move(f_move=player2.move_wolf, ks=ks, p_num=2, figure=CELL_WOLF_2,
                                          game_over=game_over, reason=reason, pool=pool, i=i)

    if debug:
        print(f'\nIteration {i} of {N_ITERATIONS}')
        ks.print_ks()
        time.sleep(slowdown)

    if reason == 'timeout1':
        ks.score1 = -1
    elif reason == 'timeout2':
        ks.score2 = -1

    print(f'p1 time remaining: {round(ks.p1_time_remaining, 4)}, p2 time remaining: {round(ks.p2_time_remaining, 4)}')

    return game_over, reason


def _get_grades(p1_score, p2_score):
    # if it's a tie, the points are distributed equally
    if p1_score == p2_score:
        p1_grade = 0.5
        p2_grade = 0.5
    elif p1_score > p2_score:
        # if sheep 2 was eaten, player 1 gets all the points
        if p2_score == -1:
            p1_grade = 1
            p2_grade = 0
        # else the winner gets 0.1, 0.9 is distributed based on points gathered
        else:
            p1_grade = 0.1 + round((0.9 * p1_score / (p1_score + p2_score)), 3)
            p2_grade = 0 + round((0.9 * p2_score / (p1_score + p2_score)), 3)
    else:
        # if sheep 1 was eaten, player 1 gets all the points
        if p1_score == -1:
            p1_grade = 0
            p2_grade = 1
        # else the winner gets 0.1, 0.9 is distributed based on points gathered
        else:
            p1_grade = 0 + round((0.9 * p1_score / (p1_score + p2_score)), 3)
            p2_grade = 0.1 + round((0.9 * p2_score / (p1_score + p2_score)), 3)

    return p1_grade, p2_grade


def _run_no_graphics(ks, player1, player2, pool):
    iterations_run = 0
    for i in range(1, N_ITERATIONS + 1):
        game_over, reason = _kingsheep_iteration(i=i, ks=ks, player1=player1, player2=player2, pool=pool, reason='')
        iterations_run += 1
        if game_over:
            print(reason)
            break
    return iterations_run


def _run_graphics(ks, player1, player2, pool):
    game = KingsheepWindow("Kingsheep", N_ITERATIONS, ks, player1, player2, pool, reason='')
    game.setup()
    arcade.run()


def _parse_game_args():
    parser = argparse.ArgumentParser(description="Run the Kingsheep Game")
    parser.add_argument("-d", "--debug", help="turn on debug mode", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int,
                        help="verbosity of the output (1: elapsed time, 2: system messages, 3: ending board")

    parser.add_argument("-p1m", "--player1module", help="name of module that defines player 1")
    parser.add_argument("-p1n", "--player1name", help="name of class that defines player 1")

    parser.add_argument("-p2m", "--player2module", help="name of module that defines player 2")
    parser.add_argument("-p2n", "--player2name", help="name of class that defines player 2")

    parser.add_argument("-g", "--graphics", help="turn on graphics based on arcade (http://arcade.academy/index.html)",
                        action="store_true")

    parser.add_argument("-s", "--slowdown", type=float,
                        help="slowdown in each iteration in seconds (fractions allowed")

    parser.add_argument("map", help="map file")
    return parser.parse_args()


def _set_global_variables(game_args):
    global debug
    if game_args.debug:
        debug = True

    global verbosity
    if game_args.verbosity:
        verbosity = game_args.verbosity

    global slowdown
    if game_args.slowdown:
        slowdown = game_args.slowdown

    global graphics
    if game_args.graphics:
        graphics = True


def _get_player1class(game_args):
    if game_args.player1module:
        mod1 = importlib.import_module(game_args.player1module)
    else:
        mod1 = importlib.import_module("random_player")

    if game_args.player1name:
        player1class = getattr(mod1, game_args.player1name)
    else:
        player1class = getattr(mod1, "RandomPlayer")

    return player1class


def _get_player2class(game_args):
    if game_args.player2module:
        mod2 = importlib.import_module(game_args.player2module)
    else:
        mod2 = importlib.import_module("random_player")

    if game_args.player2name:
        player2class = getattr(mod2, game_args.player2name)
    else:
        player2class = getattr(mod2, "RandomPlayer")

    return player2class


def _get_map_name(game_args):
    if game_args.map:
        map_name = game_args.map
    else:
        map_name = "resources/test.map"

    return map_name


def _kingsheep_play(player1class, player2class, map_name):
    if verbosity > 2:
        print('\n >>> Starting up Kingsheep\n')

    player1, player2 = player1class(), player2class()
    ks = KsField(map_name, name1=player1.name, name2=player2.name)
    pool = Pool()

    start_time = time.perf_counter()
    if graphics:
        _run_graphics(ks, player1, player2, pool)
    else:
        iterations_run = _run_no_graphics(ks, player1, player2, pool)

    elapsed_time = time.perf_counter() - start_time

    p1_grade, p2_grade = _get_grades(ks.score1, ks.score2)
    ks.grading1, ks.grading2 = p1_grade, p2_grade

    print(f'Player one got  {round(ks.grading1, 2)} points, Player two got {round(ks.grading2, 2)} points')

    if verbosity > 2 and not graphics:
        print(f'\n >>> Finishing Kingsheep after {iterations_run} iterations \n\nFinal Field:')

    ks.print_ks()
    if verbosity > 1:
        print(f'  Elapsed time: {elapsed_time}')

    pool.close()
    pool.terminate()
    pool.join()


def main():
    # command line: -p1m ksplayers -p1n PassivePlayer -p2m ksplayers -p2n KingsheepPlayer
    game_args = _parse_game_args()

    _set_global_variables(game_args=game_args)
    player1class = _get_player1class(game_args=game_args)
    player2class = _get_player2class(game_args=game_args)
    map_name = _get_map_name(game_args=game_args)

    _kingsheep_play(player1class, player2class, map_name)


debug = False
verbosity = 5
graphics = False
slowdown = 0.0

if __name__ == "__main__":
    main()
