from games import Game
from enum import Enum

# Every game must end in one of three outcomes: 
# either our AI agent wins, loses, or draws against the opponent.
Outcome = Enum("Outcome", ["WIN", "LOSS", "DRAW"])

# A utility function that simply prints out the current state of the game.         
def print_game_object(game: Game):
    print(game)