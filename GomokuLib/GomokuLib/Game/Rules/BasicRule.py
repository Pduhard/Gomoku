import re
from time import perf_counter
from GomokuLib.Game.State.GomokuState import GomokuState
import numpy as np

from GomokuLib.Game.Action import GomokuAction
from .AbstractRule import AbstractRule


class BasicRule(AbstractRule):

	restricting = True

	def get_valid(self):
		return self.engine.state.full_board ^ 1
	
	def is_valid(self, action: GomokuAction):
		ar, ac = action.action
		return np.all(self.engine.state.board[..., ar, ac] == 0)

	def winning(self, action: GomokuAction):

		# tic = perf_counter()

		state = self.engine.state
		ar, ac = action.action

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
		# print((perf_counter() - tic) * 1000)
		return False

	def count_align_this_way(self, board, x, y, dx, dy):
		xmax, ymax = self.engine.board_size
		for i in range(4):
			x += dx
			y += dy
			if (x < 0 or x >= xmax or y < 0 or y >= ymax or board[0, x, y] == 0):
				return i
		return 4
