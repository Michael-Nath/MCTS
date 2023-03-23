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
        translated_mark = self.mark_to_indicator(player.letter)
        self.board[row, col] = translated_mark
     
    def mark_to_indicator(self, mark) -> int:
        return X_MARK_INDICATOR if mark == 'X' else O_MARK_INDICATOR

    def indicator_to_mark(self, indicator) -> str:
        if indicator == X_MARK_INDICATOR:
            return 'X'
        if indicator == O_MARK_INDICATOR:
            return 'O'
        return '_'
    def get_init_game_state(self) -> np.ndarray:
        return np.full((GRID_ROWS,GRID_COLS), NO_MARK_INDICATOR)

    def get_current_game_state(self) -> np.ndarray:
        return self.board
    
    def is_terminal_state(self, state): 
        for i in range(GRID_ROWS):
            row_no_dups = np.unique(state[i])
            if NO_MARK_INDICATOR in row_no_dups:
                continue
            if len(row_no_dups) == 1:
                return (True, self.indicator_to_mark(row_no_dups[0]))

        for i in range(GRID_COLS):
            col_no_dups = np.unique(state[:, i])
            if NO_MARK_INDICATOR in col_no_dups:
                continue
            if len(col_no_dups) == 1:
                return (True, self.indicator_to_mark(col_no_dups[0]))

        unique_tl_br_diagonal = np.unique(state.diagonal())
        if len(unique_tl_br_diagonal) == 1 and NO_MARK_INDICATOR not in unique_tl_br_diagonal:
            return (True, self.indicator_to_mark(unique_tl_br_diagonal[0]))

        unique_bl_tr_diagonal = np.unique(np.fliplr(state).diagonal())
        if len(unique_bl_tr_diagonal) == 1 and NO_MARK_INDICATOR not in unique_bl_tr_diagonal:
            return (True, self.indicator_to_mark(unique_bl_tr_diagonal[0]))

        # Check if the grid is completely marked up. Control only reaches here if no row/col is dominated by a single mark. 
        if NO_MARK_INDICATOR not in state.flatten():
            return (True, '_')

        return (False, None)

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
