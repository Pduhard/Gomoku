from __future__ import annotations
from typing import Union, TYPE_CHECKING

import pygame
from pygame import event

if TYPE_CHECKING:
    from GomokuLib.Game.State.AbstractState import AbstractState

class Human:

    def __str__(self):
        return f"Human"

    def play_turn(self, engine) -> tuple[int]:
        return engine.wait_player_action()

