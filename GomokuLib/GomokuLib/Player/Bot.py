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

    def play_turn(self) -> AbstractAction:
        return self.algo(self.engine)[1]
