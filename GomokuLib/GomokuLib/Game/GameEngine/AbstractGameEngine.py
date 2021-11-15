
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer
    from ..Action.AbstractAction import AbstractAction
    from ..State.AbstractState import AbstractState


class AbstractGameEngine(metaclass=ABCMeta):

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]], **kwargs) -> None:
        for p in players:
            p.init_engine(self)

    @abstractmethod
    def get_state(self) -> AbstractState:
        pass

    @abstractmethod
    def get_actions(self) -> list[AbstractAction]:
        pass

    @abstractmethod
    def apply_action(self, action: AbstractAction) -> None:
        pass

    @abstractmethod
    def run(self) -> AbstractPlayer:
        pass

    # def is_endgame(self) -> AbstractPlayer:
    #     pass
