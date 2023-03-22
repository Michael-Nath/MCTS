from NaiveMCTS import NaiveMCTS
from games.tictactoe.TicTacToe import TicTacToe
from games.tictactoe.TicTacToePlayer import TicTacToePlayer

tictactoe_game = TicTacToe();
mcts = NaiveMCTS(tictactoe_game);
mcts.step()

# TESTING IF STEP FUNCTION WORKS
bot = TicTacToePlayer("O")
