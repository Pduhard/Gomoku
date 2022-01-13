from __future__ import annotations
from abc import ABCMeta, abstractmethod
from abc import abstractmethod
from typing import Any
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer
    from ..Action.AbstractAction import AbstractAction
    from ..State.AbstractState import AbstractState
# from GomokuLib.Game.GameEngine.Gomoku import Gomoku
# from GomokuLib.Game.Action.GomokuAction import GomokuAction
import numpy as np


class AbstractRule(metaclass=ABCMeta):

	def __init__(self, engine: Any) -> None:
		self.engine = engine

	def get_valid_actions(self) -> np.ndarray:
		pass

	def is_valid_action(self, action: Any) -> bool:
		pass
