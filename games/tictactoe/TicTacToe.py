from games.tictactoe.TicTacToePlayer import TicTacToePlayer
from games.tictactoe.Game import Game
from pprint import pprint
import numpy as np

class TicTacToe(Game):
    def __init__(self):
        # create 3 x 3 grid where 0s represent a Os and 1s represent Xs
        self.board = np.full((3,3), -1)
        super().__init__(self.board)
    
    def mark_move(self, player: TicTacToePlayer, row, col):
        translated_mark = self.translate_mark(player.letter)
        self.board[row, col] = translated_mark
     
    def translate_mark(self, mark) -> int:
        return 1 if mark == 'X' else 0

    def get_init_game_state(self) -> np.ndarray:
        return np.full((3,3), -1)

    def get_current_game_state(self) -> np.ndarray:
        return self.board
    
    def __str__(self) -> str:
        stringified_board = self.board.copy()
        stringified_board[stringified_board == 1] = "X"
        stringified_board[stringified_board == 0] = "O"
        stringified_board[stringified_board == -1] = "_"
        return np.array2string(stringified_board)
    
if __name__ == "__main__":
    board = TicTacToe()
    player_one = TicTacToePlayer("X")
    board.mark_move(player_one, 0, 0)
    pprint(board.board) 
