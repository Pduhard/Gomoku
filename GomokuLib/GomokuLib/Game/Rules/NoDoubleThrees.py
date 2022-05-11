from pickletools import uint8
from time import perf_counter
from typing import Any

import fastcore
from fastcore._rules import ffi, lib as fastcore

from GomokuLib.Game.State import GomokuState
from numba import njit
import numpy as np

from GomokuLib.Game.Action import GomokuAction

from GomokuLib.Game.GameEngine import Gomoku
from .AbstractRule import AbstractRule


class NoDoubleThrees(AbstractRule):

	restricting = True # Imply existing methods get_valid() and is_valid()
	name = 'NoDoubleThrees'

	def __init__(self, engine: Any) -> None:
		super().__init__(engine)

	def get_valid(self):
		"""
			Need to find an optimized way to compute that
		"""
		actions = np.empty_like(self.engine.state.full_board)
		for r in range(self.engine.board_size[0]):
			for c in range(self.engine.board_size[1]):
				actions[r, c] = self.is_valid(GomokuAction(r, c))
		return actions

	def is_valid(self, action: GomokuAction):

		c_board = ffi.cast("char *", self.engine.state.board.ctypes.data)
		c_full_board = ffi.cast("char *", self.engine.state.full_board.ctypes.data)
		return not fastcore.is_double_threes(c_board, c_full_board, *action.action)
	
	def create_snapshot(self):
		return {}

	def update_from_snapshot(self, snapshot):
		pass

	def copy(self, engine: Gomoku, rule: AbstractRule):
		return NoDoubleThrees(engine)
