from __future__ import annotations
from typing import TYPE_CHECKING, Union
import tkinter as tk

if TYPE_CHECKING:
    from GomokuLib.Game.GameEngine.Gomoku import Gomoku
    from GomokuLib.Player.AbstractPlayer import AbstractPlayer

class GomokuGUI(Gomoku):
    
    def __init__(self, players: Union[list[AbstractPlayer],
                 tuple[AbstractPlayer]], board_size: Union[int, tuple[int]] = 19, **kwargs) -> None:
        super().__init__(players, board_size=board_size, **kwargs)
