from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from GomokuLib.Player.AbstractPlayer import AbstractPlayer

class Human(AbstractPlayer):

    def __init__(self, verbose: dict) -> None:
        self.verbose = verbose

    def play_turn(self, state: AbstractState,
                  actions: list[AbstractAction]) -> AbstractAction:

        # game_engine.display_actions()
        pass