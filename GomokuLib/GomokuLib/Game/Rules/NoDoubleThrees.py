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

def init_ft_ident():
	FT_IDENT = np.zeros((4, 6, 2), dtype=np.bool8)

	FT_IDENT[0, 1, 0] = 1
	FT_IDENT[0, 2, 0] = 1
	FT_IDENT[0, 4, 0] = 1

	FT_IDENT[1, 1, 0] = 1
	FT_IDENT[1, 3, 0] = 1
	FT_IDENT[1, 4, 0] = 1

	FT_IDENT[2, 1, 0] = 1
	FT_IDENT[2, 2, 0] = 1
	FT_IDENT[2, 3, 0] = 1

	FT_IDENT[3, 2, 0] = 1
	FT_IDENT[3, 3, 0] = 1
	FT_IDENT[3, 4, 0] = 1

	return FT_IDENT

# @njit(parallel=True, fastmath=True)
# @njit()
def njit_is_valid(rmax, cmax, ar, ac, board, FT_IDENT):

	n_free_threes = 0
	ways = [
		(1, 1),
		(1, 0),
		(1, -1),
		(0, -1),
	]
	# print("[njit_is_valid]")
	for dr, dc in ways:

		r, c = ar - dr, ac - dc
		# print(f"\nr c dr dc: {r}, {c}, {dr}, {dc}")

		flag = 0
		i = 0
		while i < 4 and flag == 0:
			if dr != 0:
				rend, rstep = r + dr * 6, dr
			else:
				rend, rstep = r, 0

			if dc != 0:
				cend, cstep = c + dc * 6, dc
			else:
				cend, cstep = c, 0
			
			start_in_bound = r >= 0 and r < rmax and c >=0 and c < cmax
			end_in_bound = rend >= 0 and rend < rmax and cend >=0 and cend < cmax
			if start_in_bound and end_in_bound:
			
				three = np.full(4, 1, dtype=np.uint8)
				l = 0
				x, y = r, c
				while l < 6:

					boardv = board[:, x, y]
					k = 0
					while k < 4:
						if np.any(boardv != FT_IDENT[k][l]):
							three[k] = 0
						k += 1
					x += rstep
					y += cstep
					l += 1
				if np.any(three == 1):
					flag = 1

			r -= dr
			c -= dc
			i += 1
			# print(f" - n_free_threes: {n_free_threes} / flag {flag}")

		n_free_threes += flag

		if n_free_threes == 2:
			return False
	
	return True




class NoDoubleThrees(AbstractRule):

	restricting = True # Imply existing methods get_valid() and is_valid()
	name = 'NoDoubleThrees'

	def __init__(self, engine: Any) -> None:
		super().__init__(engine)
		self.FT_IDENT = init_ft_ident()

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

		ar, ac = action.action

		# rmax, cmax = self.engine.board_size
		# board = self.engine.state.board
		# old_value = board[0, ar, ac]
		# board[0, ar, ac] = 1
		# truth = njit_is_valid(rmax, cmax, ar, ac, board, self.FT_IDENT)
		# board[0, ar, ac] = old_value

		c_board = ffi.cast("char *", self.engine.state.board.ctypes.data)
		c_full_board = ffi.cast("char *", self.engine.state.full_board.ctypes.data)
		res = not fastcore.is_double_threes(c_board, c_full_board, ar, ac)

		# if res is False:
		# 	print(f"Find correct double threes")
		#
		# if bool(res) != truth:
		# 	print(f"ERROR | Is valid ? res={bool(res)} / truth={truth}")
		# 	breakpoint()

		return res
	
	def create_snapshot(self):
		return {}

	def update_from_snapshot(self, snapshot):
		pass

	def copy(self, engine: Gomoku, rule: AbstractRule):
		return NoDoubleThrees(engine)
