import numpy as np
from NaiveMCTS import NaiveMCTS
from games.tictactoe.TicTacToe import TicTacToe
from games.tictactoe.TicTacToePlayer import TicTacToePlayer
from policies.random_policy import RandomTTTPolicy

NUM_ROWS = 3
NUM_COLS = 3

tictactoe_game = TicTacToe();
mcts = NaiveMCTS(tictactoe_game, 0);
policy = RandomTTTPolicy()

# TESTING IF STEP FUNCTION WORKS
bot = TicTacToePlayer("X")

for _ in range(5):
    random_row = np.random.randint(0, NUM_ROWS)
    random_col = np.random.randint(0, NUM_COLS)
    tictactoe_game.mark_move(bot, random_row, random_col)

print(tictactoe_game)
mcts.step()
