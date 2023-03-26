import numpy as np
from NaiveMCTS import NaiveMCTS
from games.tictactoe.TicTacToe import TicTacToeBoard
from games.Player import Player
from policies.RandomPolicy import RandomTTTPolicy

NUM_ROWS = 3
NUM_COLS = 3

tictactoe_game = TicTacToeBoard()
mcts = NaiveMCTS(tictactoe_game, 0, 1, RandomTTTPolicy())
policy = RandomTTTPolicy()

# TESTING IF STEP FUNCTION WORKS
bot = Player("X")
MCTS_player = Player("O")

    
while TicTacToeBoard.is_terminal_state(tictactoe_game)[0] == False:
    # bot_action = input("Provide row, column\n").split(',')
    bot_action = policy.select_action(tictactoe_game.get_current_game_state())
    tictactoe_game.mark_move(bot, int(bot_action[0]), int(bot_action[1]))
    print(tictactoe_game)
    if TicTacToeBoard.is_terminal_state(tictactoe_game)[0]:
        break
    for _ in range(1):
        mcts.step()
    action = mcts.make_move()
    tictactoe_game.mark_move(MCTS_player, action[0], action[1])
    print(tictactoe_game)

print()
print("TICTACTOE FINAL GAME STATE:")
print(tictactoe_game)

_, winner = tictactoe_game.is_terminal_state(tictactoe_game)
if winner == 1:
    winner = "BOT"
elif winner == 0:
    winner = "MCTS"
else:
    print(f"It is a TIE!")
    exit()    
print(f"Winner is {winner}!")

