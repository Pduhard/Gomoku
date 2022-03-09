from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer
    from ..Action.AbstractAction import AbstractAction
    from ..State.AbstractState import AbstractState


class AbstractGameEngine(metaclass=ABCMeta):

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]], **kwargs) -> None:
        if players:
            for p in players:
                p.init_engine(self)

    @abstractmethod
    def get_actions(self) -> np.ndarray:
        pass

    @abstractmethod
    def apply_action(self, action: AbstractAction) -> None:
        pass

    @abstractmethod
    def run(self) -> AbstractPlayer:
        pass

    # def is_endgame(self) -> AbstractPlayer:
    #     pass



class CtrlPanel(metaclass=ABCMeta):
    """
        Bot: Logs sur les algorithm du CtrlPanel
        Human: Hints apportes par les algo du CtrlPanel

    """
    def __init__(self, player: AbstractPlayer) -> None:
        pass

    def handle_click(self, event):
        pass
