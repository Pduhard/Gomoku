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

    def __init__(self, algorithm: AbstractAlgorithm) -> None:
        self.algo = algorithm

    def __str__(self):
        return f"Bot with algo: {str(self.algo)}"

    def play_turn(self) -> AbstractAction:
        # self.policy, self.action = self.algo(self.engine)
        # return self.action
        return self.algo(self.engine)[1]
