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
MCTS_player = TicTacToePlayer("O")

    
while tictactoe_game.is_terminal_state(tictactoe_game.get_current_game_state())[0] == False:
    bot_action = policy.select_action(tictactoe_game.get_current_game_state())
    tictactoe_game.mark_move(bot, bot_action[0], bot_action[1])
    print(tictactoe_game)
    if tictactoe_game.is_terminal_state(tictactoe_game.get_current_game_state())[0]:
        break
    for _ in range(10000):
        mcts.step()
    action = mcts.make_move()
    tictactoe_game.mark_move(MCTS_player, action[0], action[1])
    print(tictactoe_game)

print()
print("TICTACTOE FINAL GAME STATE:")
print(tictactoe_game)

_, winner = tictactoe_game.is_terminal_state(tictactoe_game.get_current_game_state())
if winner == 1:
    winner = "BOT"
elif winner == 0:
    winner = "MCTS"
else:
    print(f"It is a TIE!")
    exit()    
print(f"Winner is {winner}!")

