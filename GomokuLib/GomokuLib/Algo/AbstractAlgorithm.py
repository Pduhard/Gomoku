from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict

if TYPE_CHECKING:
    from GomokuLib.Game.GameEngine.AbstractGameEngine import AbstractGameEngine

from abc import abstractmethod, abstractproperty, ABCMeta



class AbstractAlgorithm(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, game_engine: AbstractGameEngine) -> Dict[tuple[int], float]:
        pass