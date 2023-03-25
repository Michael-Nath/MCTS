

class Policy:
    def __init__(self) -> None:
        raise NotImplementedError
    def select_action(self, state):
        raise NotImplementedError 