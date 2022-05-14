from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from Game.GameEngine.AbstractGameEngine import AbstractGameEngine
    from Game.State.GomokuState import GomokuState
    from Player.AbstractPlayer import AbstractPlayer

__all__ = [
    'annotations',
    'Union',
    'AbstractGameEngine',
    'GomokuState',
    'AbstractPlayer'
]