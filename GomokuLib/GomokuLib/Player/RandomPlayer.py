
import numpy as np

class RandomPlayer:

    def __init__(self, verbose: dict = None) -> None:
        self.verbose = verbose or {}

    def __str__(self):
        return f"Random player"

    def init(self):
        pass

    def play_turn(self, runner) -> tuple[int]:

        actions = runner.engine.get_actions()
        id = np.random.choice(
            361,
            p=actions.flatten() / np.count_nonzero(actions)
        )
        return id // 19, id % 19
