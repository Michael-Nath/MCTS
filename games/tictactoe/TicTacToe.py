from games.tictactoe.TicTacToePlayer import TicTacToePlayer
from games.tictactoe.Game import Game
from pprint import pprint
import numpy as np

X_MARK_INDICATOR = 1
O_MARK_INDICATOR = 0
NO_MARK_INDICATOR = -1
GRID_ROWS = 3
GRID_COLS = 3 

class TicTacToe(Game):
    def __init__(self):
        self.board = np.full((GRID_ROWS,GRID_COLS), NO_MARK_INDICATOR)
        super().__init__(self.board)
    
    def mark_move(self, player: TicTacToePlayer, row, col):
        translated_mark = self.translate_mark(player.letter)
        self.board[row, col] = translated_mark
     
    def translate_mark(self, mark) -> int:
        return X_MARK_INDICATOR if mark == 'X' else O_MARK_INDICATOR

    def get_init_game_state(self) -> np.ndarray:
        return np.full((GRID_ROWS,GRID_COLS), NO_MARK_INDICATOR)

    def get_current_game_state(self) -> np.ndarray:
        return self.board
    
    def is_terminal_state(self, state) -> bool:
        # Check if any of the rows are filled with same mark. 
        unique_row_counts = [len(np.unique(row)) for row in state if NO_MARK_INDICATOR not in row]
        if 1 in unique_row_counts:
            return True
        unique_col_counts = [len(np.unique(col)) for col in state.T if NO_MARK_INDICATOR not in col]
        if 1 in unique_col_counts:
            return True
        unique_tl_br_diagonal = np.unique(state.diagonal())
        if len(unique_tl_br_diagonal) == 1 and NO_MARK_INDICATOR not in unique_tl_br_diagonal:
            return True
        unique_bl_tr_diagonal = np.fliplr(state).diagonal()
        if len(unique_bl_tr_diagonal) == 1 and NO_MARK_INDICATOR not in unique_bl_tr_diagonal:
            return True

        return False

    def __str__(self) -> str:
        stringified_board = self.board.copy().astype("object")
        stringified_board[stringified_board == X_MARK_INDICATOR] = "X"
        stringified_board[stringified_board == O_MARK_INDICATOR] = "O"
        stringified_board[stringified_board == NO_MARK_INDICATOR] = "_"
        return np.array2string(stringified_board)
    
if __name__ == "__main__":
    board = TicTacToe()
    player_one = TicTacToePlayer("X")
    board.mark_move(player_one, 0, 0)
    pprint(board.board) 
