from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from abc import abstractmethod, abstractproperty, ABCMeta



class AbstractAlgorithm(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        pass

    def __call__(self, state: AbstractState, actions: list[AbstractAction]) -> list[AbstractAction]:
        res = self.run(state, actions)
        if isinstance(res, list):
            return sorted(res, key=lambda x: x.score)
        else:
            return [res]

    @abstractmethod
    def run(self, state: AbstractState, actions: list[AbstractAction]) -> Union[list[AbstractAction], AbstractAction]:
        pass