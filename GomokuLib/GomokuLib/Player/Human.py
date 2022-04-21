from __future__ import annotations
from typing import Union, TYPE_CHECKING

import pygame
from pygame import event

if TYPE_CHECKING:
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from GomokuLib.Player.AbstractPlayer import AbstractPlayer

class Human(AbstractPlayer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"Human"

    def play_turn(self) -> AbstractAction:
        return self.engine.wait_player_action()

