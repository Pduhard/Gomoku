from time import perf_counter
from typing import Any
from GomokuLib.Game.Action import GomokuAction
import numpy as np
from pygame import mask

from GomokuLib.Game.GameEngine import Gomoku
from .AbstractRule import AbstractRule

def init_capture_mask():
	CAPTURE_MASK = np.zeros((8, 2, 7, 7), dtype=np.int32)

	CAPTURE_MASK[0, 0, 0, 0] = 1
	CAPTURE_MASK[0, 1, 1, 1] = 1
	CAPTURE_MASK[0, 1, 2, 2] = 1

	CAPTURE_MASK[1, 0, 0, 3] = 1
	CAPTURE_MASK[1, 1, 1, 3] = 1
	CAPTURE_MASK[1, 1, 2, 3] = 1

	CAPTURE_MASK[2, 0, 0, 6] = 1
	CAPTURE_MASK[2, 1, 1, 5] = 1
	CAPTURE_MASK[2, 1, 2, 4] = 1

	CAPTURE_MASK[3, 0, 3, 6] = 1
	CAPTURE_MASK[3, 1, 3, 5] = 1
	CAPTURE_MASK[3, 1, 3, 4] = 1

	CAPTURE_MASK[4, 0, 6, 6] = 1
	CAPTURE_MASK[4, 1, 5, 5] = 1
	CAPTURE_MASK[4, 1, 4, 4] = 1

	CAPTURE_MASK[5, 0, 6, 3] = 1
	CAPTURE_MASK[5, 1, 5, 3] = 1
	CAPTURE_MASK[5, 1, 4, 3] = 1

	CAPTURE_MASK[6, 0, 6, 0] = 1
	CAPTURE_MASK[6, 1, 5, 1] = 1
	CAPTURE_MASK[6, 1, 4, 2] = 1

	CAPTURE_MASK[7, 0, 3, 0] = 1
	CAPTURE_MASK[7, 1, 3, 1] = 1
	CAPTURE_MASK[7, 1, 3, 2] = 1

	# CAPTURE_MASK = CAPTURE_MASK[]

	return CAPTURE_MASK

class Capture(AbstractRule):

	def __init__(self, engine: Any) -> None:
		super().__init__(engine)
		self.CAPTURE_MASK=init_capture_mask()
		self.player_count_capture = [0, 0]

	def endturn(self, action: GomokuAction):
		
		state = self.engine.state
		ar, ac = action.action
		tic = perf_counter()

		# Get 7x7-subboard bounds around last action
		row_start = max(ar - 3, 0)
		relrow_start = 3 - (ar - row_start)

		row_end = min(ar + 4, self.engine.board_size[0])
		relrow_end = relrow_start + (row_end - row_start)

		col_start = max(ac - 3, 0)
		relcol_start = 3 - (ac - col_start)

		col_end = min(ac + 4, self.engine.board_size[1])
		relcol_end = relcol_start + (col_end - col_start)

		# Padd subboard to 7x7 dim
		sub_board = np.zeros((2, 7, 7), dtype=np.int32)
		sub_board[..., relrow_start : relrow_end, relcol_start : relcol_end] = state.board[..., row_start : row_end, col_start : col_end]

		# Keep align stone around action in each direction
		captures = self.CAPTURE_MASK & sub_board

		# Count align stone per line
		captures = np.sum(captures.reshape(8, 2 * 7 * 7), axis=-1)

		# If 3 stones are align, capture is present
		capture_flag = captures == 3

		capture_count = np.count_nonzero(capture_flag)
		if capture_count > 0:
			sm = np.add.reduce(self.CAPTURE_MASK[capture_flag, 1, ...], axis=0)
			state.board[..., 1, row_start : row_end, col_start : col_end] ^= sm[..., relrow_start : relrow_end, relcol_start : relcol_end]
			self.player_count_capture[self.engine.player_idx] += capture_count

		# print((self.CAPTURE_MASK & sub_board).shape)
		# print(sub_board.shape, self.CAPTURE_MASK.shape)
		# print(sub_board[0], sub_board[1], sep ='\n\n')
		# print()

		# capture_count =\
		# 	sum(
		# 		sum(
		# 			self.capture(state.board, ar, ac, i, j)
		# 			for i in range(-1, 2)
		# 			if self.is_capture_this_way(state.board, ar, ac, i, j)
		# 		)
		# 		for j in range(-1, 2)
		# 	)
		# self.player_count_capture[self.engine.player_idx] += capture_count

		# print((perf_counter() - tic) * 1000, self.player_count_capture)

	# def capture(self, board, x, y, dx, dy):
	# 	board[1, x + dx, y + dy] = 0
	# 	board[1, x + dx * 2, y + dy * 2] = 0
	# 	return 1
	# def is_capture_this_way(self, board, x, y, dx, dy):
	# 	return board[1, x + dx, y + dy]\
	# 	   and board[1, x + dx * 2, y + dy * 2]\
	# 	   and board[0, x + dx * 3, y + dy * 3]

	def winning(self, action: GomokuAction):
		return self.player_count_capture[self.engine.player_idx] >= 5

	def copy(self, engine: Gomoku):
		rule = Capture(engine)
		rule.player_count_capture = self.player_count_capture
		return rule
