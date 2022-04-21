import random

import numpy as np

from GomokuLib.Player.AbstractPlayer import AbstractPlayer
from GomokuLib.Game.State.AbstractState import AbstractState
from GomokuLib.Game.Action.GomokuAction import GomokuAction


class RandomPlayer(AbstractPlayer):

    def __init__(self, verbose: dict = None) -> None:
        self.verbose = verbose or {}

    def __str__(self):
        return f"Random player"

    def play_turn(self) -> GomokuAction:

        actions = self.engine.get_actions()
        id = np.random.choice(self.engine.board_size[0] * self.engine.board_size[1], p=actions.flatten()/np.count_nonzero(actions))
        print(f"RandomPlayer id {id}")

        gaction = GomokuAction(id // self.engine.board_size[1], id % self.engine.board_size[1])
        # print(f"RandomPlayer choose {gaction}")
        return gaction
