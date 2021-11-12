
from abc import ABCMeta, abstractmethod
from typing import Union
import GomokuLib
from GomokuLib import AbstractAction
from GomokuLib import AbstractState
from GomokuLib import AbstractPlayer

class AbstractGameEngine(metaclass=ABCMeta):

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]],
                 board_size: Union[int, tuple[int]] = 19, **kwargs) -> None:
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
