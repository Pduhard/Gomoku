from time import perf_counter
from typing import Any
from GomokuLib.Game.State import GomokuState
import numpy as np

from GomokuLib.Game.Action import GomokuAction
from .AbstractRule import AbstractRule

def init_ft_ident():
	FT_IDENT = np.zeros((4, 6, 2))

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
	

class NoDoubleThrees(AbstractRule):

	restricting = True

	def __init__(self, engine: Any) -> None:
		super().__init__(engine)
		self.FT_IDENT = init_ft_ident()


	def get_valid(self):
		return np.ones_like(self.engine.state.full_board)
	
	def is_valid(self, action: GomokuAction):
		
		tic = perf_counter()
		ar, ac = action.action

		board = self.engine.state.board.copy()
		board[0, ar, ac] = 1  # c'est immonde mais bon

		free_threes = self.count_free_threes(board, ar - 1, ac - 1, 1, 1)
		free_threes += self.count_free_threes(board, ar - 1, ac, 1, 0)
		if free_threes == 2:
			return False
		free_threes += self.count_free_threes(board, ar - 1, ac + 1, 1, -1)
		if free_threes == 2:
			return False
		# elif free_threes ==0:
		# 	return True
		free_threes += self.count_free_threes(board, ar, ac + 1, 0, -1)
		print('bousin:', (perf_counter() - tic) * 1000)
		return free_threes < 2
	
	def count_free_threes(self, board, x, y, dx, dy):
		"""
			A free-three is an alignement of three stones that, if not immediately blocked,
			allows for an indefendable alignment of four stones
			(that’s to say an alignment of four stones with two unobstructed extremities)
		"""
		rxmax, rymax = self.engine.board_size
		for i in range(4):
			align = []

			# for rx, ry in zip(
			# 	range(max(x, 0), min(x + dx * 6, self.engine.board_size[0]), dx),
			# 	range(max(y, 0), min(y + dy * 6, self.engine.board_size[1]), dy)
			# ):
			iter_x = range(x, x + dx * 6, dx) if dx != 0 else [x] * 6
			iter_y = range(y, y + dy * 6, dy) if dy != 0 else [y] * 6
			for rx, ry in zip(iter_x, iter_y):
				if (rx >= 0 and rx < rxmax and ry >= 0 and ry < rymax):
					align.append(board[:, rx, ry])
				# else:
				# 	align.append([0, 0])
			align = np.array(align)
			if len(align) == 6:
				if np.all(align == self.FT_IDENT[0]):
					return 1
				elif np.all(align == self.FT_IDENT[1]):
					return 1
				elif np.all(align == self.FT_IDENT[2]):
					return 1
				elif np.all(align == self.FT_IDENT[3]):
					return 1
			# print(align)
			x -= dx
			y -= dy
			# exit(0)
		# fthrees = board[..., x : x+dx*6 : dx, y : y+dy*6 : dy]
		# print(fthrees)
		# exit(0)

		return 0