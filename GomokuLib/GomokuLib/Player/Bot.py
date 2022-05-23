from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from GomokuLib.Algo.AbstractAlgorithm import AbstractAlgorithm


class Bot:

    def __init__(self, algorithm: AbstractAlgorithm) -> None:
        self.algo = algorithm

    def __str__(self):
        return f"Bot with algo: {str(self.algo)}"

    def play_turn(self, runner) -> tuple[int]:
        return self.algo(runner.engine)[1]
