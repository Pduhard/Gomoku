from GomokuLib.Game.Action import GomokuAction

from GomokuLib.Game.GameEngine import Gomoku
from .AbstractRule import AbstractRule
from fastcore._rules import ffi, lib as fastcore

from numba import njit


class BasicRule(AbstractRule):

	name = 'BasicRule'
	restricting = True  # Imply existing methods get_valid() and is_valid()

	def get_valid(self):
		return self.engine.state.full_board ^ 1

	def is_valid(self, action: GomokuAction):
		# return njit_is_valid(self.engine.state.full_board, *action.action)
		return self.engine.state.full_board[action.action] == 0

	def winning(self, action: GomokuAction):

		# win = njit_is_align(self.engine.state.board, *action.action)

		ar, ac = action.action
		gz = self.engine.get_game_zone()
		c_board = ffi.cast("char *", self.engine.state.board.ctypes.data)
		win = fastcore.is_winning(c_board, 0, ar, ac, gz[0], gz[1], gz[2], gz[3])
		return win

	def create_snapshot(self):
		return {}

	def update_from_snapshot(self, snapshot):
		pass

	def update(self, engine: Gomoku, rule: AbstractRule):
		return BasicRule(engine)


# @njit()
# def njit_is_align(board, ar, ac, p_id: int = 0, n_align: int = 5):
#
# 	branch_align = n_align - 1
# 	ways = [
# 		(-1, -1),
# 		(-1, 0),
# 		(-1, 1),
# 		(0, -1)
# 	]
#
# 	# 4 direction
# 	for rway, cway in ways:
#
# 		# Slide 4 times
# 		r1, c1 = ar, ac
# 		count1 = branch_align
# 		i = 0
# 		while (i < branch_align and count1 == branch_align):
#
# 			r1 += rway
# 			c1 += cway
# 			if (r1 < 0 or r1 >= 19 or c1 < 0 or c1 >= 19 or board[p_id, r1, c1] == 0):
# 				count1 = i
# 			i += 1
#
# 		r2, c2 = ar, ac
# 		count2 = branch_align
# 		i = 0
# 		while (i < branch_align and count2 == branch_align):
# 			r2 -= rway
# 			c2 -= cway
# 			if (r2 < 0 or r2 >= 19 or c2 < 0 or c2 >= 19 or board[p_id, r2, c2] == 0):
# 				count2 = i
# 			i += 1
#
# 		# print(f"dir {rway} {cway}: {count1} + {count2} + 1")
# 		if (count1 + count2 + 1 >= n_align):
# 			return True
#
# 	return False
