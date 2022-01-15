from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict

if TYPE_CHECKING:
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from abc import abstractmethod, abstractproperty, ABCMeta



class AbstractAlgorithm(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, state: AbstractState, actions: list[AbstractAction]) -> Dict[AbstractAction, float]:
        pass

    # def __call__(self, state: AbstractState, actions: list[AbstractAction]) -> Dict[AbstractAction, float]:
    #     res = self.run(state, actions)
    #     assert len(res) == len(actions)
    #     return {a: r for a, r in zip(actions, res)}

    # def run(self, state: AbstractState, actions: list[AbstractAction]) -> list[AbstractAction]:
    # """
    #     Run() have to score respectively each actions 
    # """
    #     pass