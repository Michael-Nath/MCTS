from games.tictactoe.TicTacToePlayer import TicTacToePlayer
from games.Player import Player
from games.Game import Game
from typing import Union
from pprint import pprint
import numpy as np

X_MARK_INDICATOR = 1
O_MARK_INDICATOR = 0
NO_MARK_INDICATOR = -1
GRID_ROWS = 3
GRID_COLS = 3 

class TicTacToeBoard(Game):
    def __init__(self, configuration:Union[np.ndarray, None] = None):
        if configuration is None:
            self.board = np.full((GRID_ROWS,GRID_COLS), NO_MARK_INDICATOR)
        else:
            self.board = configuration
        super().__init__(self.board)
    
    def mark_move(self, player: Player, row, col):
        translated_mark = TicTacToeBoard.mark_to_indicator(player.mark)
        self.board[row, col] = translated_mark
    
    
     
    @staticmethod 
    def mark_to_indicator(mark) -> int:
        return X_MARK_INDICATOR if mark == 'X' else O_MARK_INDICATOR
    
    @staticmethod
    def indicator_to_mark(indicator) -> str:
        if indicator == X_MARK_INDICATOR:
            return 'X'
        if indicator == O_MARK_INDICATOR:
            return 'O'
        return '_'

    def copy_(self) -> Game:
        return TicTacToeBoard(self.board.copy())
    
    @staticmethod
    def get_init_game_state() -> np.ndarray:
        return np.full((GRID_ROWS,GRID_COLS), NO_MARK_INDICATOR)

    def get_current_game_state(self) -> np.ndarray:
        return self.board
    
    def get_next_game_state(self, action, mark):
        '''
        Returns the next game state (s') from the current state (s) after taking
        `action`, and marked with `mark`
        '''
        new_state = self.board.copy()
        new_state[tuple(action)] = mark
        return TicTacToeBoard(new_state)
    
    def get_next_game_states(self, mark):
        '''
        Returns all reachable game states from given state, and marked with `mark`
        '''
        pos_next_states = []
        input_actions = self.get_all_next_actions()
        for action in input_actions:
            new_board_obj = self.get_next_game_state(action, mark)
            pos_next_states.append(new_board_obj)
            input_actions.append(action)
        return pos_next_states, input_actions
    
    def get_all_next_actions(self):
        pos_indices = np.where(self.board == NO_MARK_INDICATOR)
        return np.array(list(zip(pos_indices))).reshape(-1, 2)
     
    @staticmethod
    def is_terminal_state(game_obj: Game): 
        for i in range(GRID_ROWS):
            row_no_dups = np.unique(game_obj.state[i])
            if NO_MARK_INDICATOR in row_no_dups:
                continue
            if len(row_no_dups) == 1:
                return (True, row_no_dups[0])

        for i in range(GRID_COLS):
            col_no_dups = np.unique(game_obj.state[:, i])
            if NO_MARK_INDICATOR in col_no_dups:
                continue
            if len(col_no_dups) == 1:
                return (True, col_no_dups[0])

        unique_tl_br_diagonal = np.unique(game_obj.state.diagonal())
        if len(unique_tl_br_diagonal) == 1 and NO_MARK_INDICATOR not in unique_tl_br_diagonal:
            return (True, unique_tl_br_diagonal[0])

        unique_bl_tr_diagonal = np.unique(np.fliplr(game_obj.state).diagonal())
        if len(unique_bl_tr_diagonal) == 1 and NO_MARK_INDICATOR not in unique_bl_tr_diagonal:
            return (True, unique_bl_tr_diagonal[0])

        # Check if the grid is completely marked up. Control only reaches here if no row/col is dominated by a single mark. 
        if NO_MARK_INDICATOR not in game_obj.state.flatten():
            return (True, -1)

        return (False, None)

    @staticmethod
    def pretty_print_grid(grid):
        stringified_board = grid.copy().astype("object")
        stringified_board[stringified_board == X_MARK_INDICATOR] = "X"
        stringified_board[stringified_board == O_MARK_INDICATOR] = "O"
        stringified_board[stringified_board == NO_MARK_INDICATOR] = "_" 
        print(np.array2string(stringified_board))
    
    def __str__(self) -> str:
        stringified_board = self.board.copy().astype("object")
        stringified_board[stringified_board == X_MARK_INDICATOR] = "X"
        stringified_board[stringified_board == O_MARK_INDICATOR] = "O"
        stringified_board[stringified_board == NO_MARK_INDICATOR] = "_"
        return np.array2string(stringified_board)
    
if __name__ == "__main__":
    board = TicTacToeBoard()
    player_one = TicTacToePlayer("X")
    board.mark_move(player_one, 0, 0)
    pprint(board.board) 
