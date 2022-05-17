from __future__ import annotations
from typing import Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from GomokuLib.Algo.AbstractAlgorithm import AbstractAlgorithm
    from GomokuLib.Game.State.AbstractState import AbstractState

    # 'AbstractGameEngine',
class Bot:

    def __init__(self, algorithm: AbstractAlgorithm) -> None:
        self.algo = algorithm

    def __str__(self):
        return f"Bot with algo: {str(self.algo)}"

    def play_turn(self, engine, **kwargs) -> tuple[int]:
        return self.algo(engine)[1]
