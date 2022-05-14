from __future__ import annotations
from abc import ABCMeta, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Game.GameEngine.AbstractGameEngine import AbstractGameEngine
    from ..Game.State.AbstractState import AbstractState

class AbstractPlayer(metaclass=ABCMeta):

    engine: AbstractGameEngine = None

    @abstractmethod
    def __init__(self) -> None:
        pass

    def init_engine(self, engine: AbstractGameEngine) -> None:
        self.engine = engine

    @abstractmethod
    def play_turn(self) -> tuple[int]:
        pass
