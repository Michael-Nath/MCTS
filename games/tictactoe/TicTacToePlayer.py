class TicTacToePlayer:
    def __init__(self, letter) -> None:
        assert letter in ['X', 'O']
        self.letter = letter
