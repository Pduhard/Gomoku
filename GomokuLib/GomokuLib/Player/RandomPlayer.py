
import numpy as np

class RandomPlayer:

    def __init__(self, verbose: dict = None) -> None:
        self.verbose = verbose or {}

    def __str__(self):
        return f"Random player"

    def play_turn(self, runner) -> tuple[int]:

        engine = runner.engine
        actions = engine.get_actions()
        id = np.random.choice(engine.board_size[0] * engine.board_size[1], p=actions.flatten()/np.count_nonzero(actions))
        print(f"RandomPlayer id {id}")

        gaction = (id // engine.board_size[1], id % engine.board_size[1])
        # print(f"RandomPlayer choose {gaction}")
        return gaction
