from __future__ import annotations
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from GomokuLib.Algo.AbstractAlgorithm import AbstractAlgorithm
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from GomokuLib.Player.AbstractPlayer import AbstractPlayer

class Bot(AbstractPlayer):

    def __init__(self, verbose: dict, algorithm: AbstractAlgorithm) -> None:
        self.algo = algorithm
        self.verbose = verbose

    def play_turn(self, state: AbstractState,
                  actions: list[AbstractAction]) -> AbstractAction:
        res = self.algo(state, actions)
        return res[0]
