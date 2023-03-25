from games.tictactoe import Game
from enum import Enum

Outcome = Enum("Outcome", ["WIN", "LOSS", "DRAW"])

# A utility function that simply prints out the current state of the game.         
def print_game_object(game: Game):
    print(game)