from __future__ import annotations
from typing import Union, TYPE_CHECKING

from GomokuLib.Algo.AbstractAlgorithm import AbstractAlgorithm
from GomokuLib.Algo.MCTSParallel import MCTSParallel


class Bot:

    def __init__(self, algorithm: AbstractAlgorithm) -> None:
        self.algo = algorithm

        if isinstance(self.algo, MCTSParallel):
            self.play_turn = self._play_turn_tuple
        else:
            self.play_turn = self._play_turn_gAction

    def __str__(self):
        return f"Bot with algo: {str(self.algo)}"

    def _play_turn_gAction(self, engine, **kwargs):
        return self.algo(engine)[1]

    def _play_turn_tuple(self, engine, **kwargs):
        return self.algo(engine)
