from __future__ import annotations
from abc import ABCMeta, abstractmethod
from abc import abstractmethod
from typing import Any
from typing import Union, TYPE_CHECKING

from GomokuLib.Game.GameEngine import Gomoku

if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer
    from ..Action.AbstractAction import AbstractAction
    from ..State.AbstractState import AbstractState
# from GomokuLib.Game.GameEngine.Gomoku import Gomoku
# from GomokuLib.Game.Action.GomokuAction import GomokuAction
import numpy as np


class AbstractRule(metaclass=ABCMeta):
	"""
		Caro rule: Must have an overline (length >= 6)  or  an unbroken row of 5 stones that is not blocked at either end
	"""

	def __init__(self, engine: Any) -> None:
		self.engine = engine

	# def opening(self):
	# 	pass

	# def restricting(self):
	# 	pass

	# def endturn(self):
	# 	pass

	# def winning(self):
	# 	pass

	@abstractmethod
	def update(self, engine: Gomoku, rule: AbstractRule):
		pass

	def update_from_snapshot(self, snapshot):
		pass
