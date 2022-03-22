from __future__ import annotations
from typing import Union, TYPE_CHECKING

import numpy as np

from GomokuLib.Game.Action import GomokuAction


if TYPE_CHECKING:
    from GomokuLib.Algo.AbstractAlgorithm import AbstractAlgorithm
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from GomokuLib.Player.AbstractPlayer import AbstractPlayer

class Bot(AbstractPlayer):

    def __init__(self, algorithm: AbstractAlgorithm, verbose: dict = None) -> None:
        self.algo = algorithm
        self.verbose = verbose
        self.last_policy = None

    def play_turn(self) -> AbstractAction:

        self.last_policy = self.algo(self.engine)
        best_action_idx = np.argmax(self.last_policy)
        print("policy (rewards sum / visit count):\n", self.last_policy)
        print("best arg:\n", np.argmax(self.last_policy))
        return GomokuAction(
            best_action_idx // self.engine.board_size[1],
            best_action_idx % self.engine.board_size[1]
        )

    def get_last_policy(self):
        return self.last_policy
