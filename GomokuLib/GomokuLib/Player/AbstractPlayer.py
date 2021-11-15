from __future__ import annotations
from abc import ABCMeta, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Game.GameEngine.AbstractGameEngine import AbstractGameEngine
    from ..Game.State.AbstractState import AbstractState
    from ..Game.Action.AbstractAction import AbstractAction

class AbstractPlayer(metaclass=ABCMeta):

    engine: AbstractGameEngine = None

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def play_turn(self, state: AbstractState,
                  actions: list[AbstractAction]) -> AbstractAction:
        pass

    def init_engine(self, engine: AbstractGameEngine) -> None:
        self.engine = engine
