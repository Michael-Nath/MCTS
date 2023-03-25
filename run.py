import numpy as np
from NaiveMCTS import NaiveMCTS
from games.tictactoe.TicTacToe import TicTacToe
from games.tictactoe.TicTacToePlayer import TicTacToePlayer
from policies.RandomPolicy import RandomTTTPolicy

NUM_ROWS = 3
NUM_COLS = 3

tictactoe_game = TicTacToe();
mcts = NaiveMCTS(tictactoe_game, 0, 1, RandomTTTPolicy());
policy = RandomTTTPolicy()

# TESTING IF STEP FUNCTION WORKS
bot = TicTacToePlayer("X")

for _ in range(1):
    random_row = np.random.randint(0, NUM_ROWS)
    random_col = np.random.randint(0, NUM_COLS)
    tictactoe_game.mark_move(bot, random_row, random_col)

print(tictactoe_game)
for _ in range(5):
    mcts.step()
print(mcts)
