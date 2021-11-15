from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from .AbstractAlgorithm import AbstractAlgorithm

class   MCTS(AbstractAlgorithm):

    def __init__(self, state: AbstractState, actions: AbstractAction) -> None:
        super().__init__()

    def run(self, state: AbstractState, actions: list[AbstractAction]) -> Union[list[AbstractAction], AbstractAction]:
        return actions
