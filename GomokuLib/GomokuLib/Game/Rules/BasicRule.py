from pickle import FALSE
import re
from time import perf_counter
from GomokuLib.Game.State.GomokuState import GomokuState
from numba import njit
import numpy as np
from GomokuLib.Game.Action import GomokuAction

from GomokuLib.Game.GameEngine import Gomoku
from .AbstractRule import AbstractRule


# @njit()
def njit_is_align(board, ar, ac, rmax, cmax, p_id: int = 0, n_align: int = 5):

	branch_align = n_align - 1
	ways = [
		(-1, -1),
		(-1, 0),
		(-1, 1),
		(0, -1)
	]

	# 4 direction
	for rway, cway in ways:

		# Slide 4 times
		r1, c1 = ar, ac
		count1 = branch_align
		i = 0
		while (i < branch_align and count1 == branch_align):

			r1 += rway
			c1 += cway
			if (r1 < 0 or r1 >= rmax or c1 < 0 or c1 >= cmax or board[p_id, r1, c1] == 0):
				count1 = i
			i += 1

		r2, c2 = ar, ac
		count2 = branch_align
		i = 0
		while (i < branch_align and count2 == branch_align):
			r2 -= rway
			c2 -= cway
			if (r2 < 0 or r2 >= rmax or c2 < 0 or c2 >= cmax or board[p_id, r2, c2] == 0):
				count2 = i
			i += 1

		# print(f"dir {rway} {cway}: {count1} + {count2} + 1")
		if (count1 + count2 + 1 >= n_align):
			return True

	return False


class BasicRule(AbstractRule):

	name = 'BasicRule'
	restricting = True  # Imply existing methods get_valid() and is_valid()

	def get_valid(self):
		return self.engine.state.full_board ^ 1
	
	def is_valid(self, action: GomokuAction):
		ar, ac = action.action
		return np.all(self.engine.state.board[..., ar, ac] == 0)

	def winning(self, action: GomokuAction):

		ar, ac = action.action
		rmax, cmax = self.engine.board_size
		return njit_is_align(self.engine.state.board, ar, ac, rmax, cmax)
		# tic = perf_counter()


		if (self.count_align_this_way(state.board, ar, ac, -1, -1) +\
			self.count_align_this_way(state.board, ar, ac, 1, 1) + 1 >= 5):
			# print((perf_counter() - tic) * 1000)
			return True

		if (self.count_align_this_way(state.board, ar, ac, -1, 0) +\
			self.count_align_this_way(state.board, ar, ac, 1, 0) + 1 >= 5):
			# print((perf_counter() - tic) * 1000)
			return True

		if (self.count_align_this_way(state.board, ar, ac, -1, 1) +\
			self.count_align_this_way(state.board, ar, ac, 1, -1) + 1 >= 5):
			# print((perf_counter() - tic) * 1000)
			return True

		if (self.count_align_this_way(state.board, ar, ac, 0, -1) +\
			self.count_align_this_way(state.board, ar, ac, 0, 1) + 1 >= 5):
			# print((perf_counter() - tic) * 1000)
			return True
		# print("No win")
		# print((perf_counter() - tic) * 1000)
		return False

	def count_align_this_way(self, board, x, y, dx, dy):
		rmax, cmax = self.engine.board_size
		for i in range(4):
			x += dx
			y += dy
			if (x < 0 or x >= rmax or y < 0 or y >= cmax or board[0, x, y] == 0):
				return i
		return 4

	def create_snapshot(self):
		return {}

	def update_from_snapshot(self, snapshot):
		pass

	def copy(self, engine: Gomoku, rule: AbstractRule):
		return BasicRule(engine)
