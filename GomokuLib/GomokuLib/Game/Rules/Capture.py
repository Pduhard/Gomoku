from time import perf_counter
from typing import Any

import fastcore
from fastcore._rules import ffi, lib as fastcore

from GomokuLib.Game.Action import GomokuAction
import numpy as np
# from numba import njit

from GomokuLib.Game.GameEngine import Gomoku

from GomokuLib.Game.Rules.GameEndingCapture import ForceWinPlayer
from .AbstractRule import AbstractRule

# def init_capture_mask():
# 	CAPTURE_MASK = np.zeros((8, 2, 7, 7), dtype=np.bool8)
#
# 	CAPTURE_MASK[0, 0, 0, 0] = 1
# 	CAPTURE_MASK[0, 1, 1, 1] = 1
# 	CAPTURE_MASK[0, 1, 2, 2] = 1
#
# 	CAPTURE_MASK[1, 0, 0, 3] = 1
# 	CAPTURE_MASK[1, 1, 1, 3] = 1
# 	CAPTURE_MASK[1, 1, 2, 3] = 1
#
# 	CAPTURE_MASK[2, 0, 0, 6] = 1
# 	CAPTURE_MASK[2, 1, 1, 5] = 1
# 	CAPTURE_MASK[2, 1, 2, 4] = 1
#
# 	CAPTURE_MASK[3, 0, 3, 6] = 1
# 	CAPTURE_MASK[3, 1, 3, 5] = 1
# 	CAPTURE_MASK[3, 1, 3, 4] = 1
#
# 	CAPTURE_MASK[4, 0, 6, 6] = 1
# 	CAPTURE_MASK[4, 1, 5, 5] = 1
# 	CAPTURE_MASK[4, 1, 4, 4] = 1
#
# 	CAPTURE_MASK[5, 0, 6, 3] = 1
# 	CAPTURE_MASK[5, 1, 5, 3] = 1
# 	CAPTURE_MASK[5, 1, 4, 3] = 1
#
# 	CAPTURE_MASK[6, 0, 6, 0] = 1
# 	CAPTURE_MASK[6, 1, 5, 1] = 1
# 	CAPTURE_MASK[6, 1, 4, 2] = 1
#
# 	CAPTURE_MASK[7, 0, 3, 0] = 1
# 	CAPTURE_MASK[7, 1, 3, 1] = 1
# 	CAPTURE_MASK[7, 1, 3, 2] = 1
#
# 	return CAPTURE_MASK
#
# # @njit()
# def njit_endturn(board, action, board_size, CAPTURE_MASK):
#
# 	ar, ac = action
# 	# Get 7x7-subboard bounds around last action
# 	row_start = max(ar - 3, 0)
# 	relrow_start = 3 - (ar - row_start)
#
# 	row_end = min(ar + 4, board_size[0])
# 	relrow_end = relrow_start + (row_end - row_start)
#
# 	col_start = max(ac - 3, 0)
# 	relcol_start = 3 - (ac - col_start)
#
# 	col_end = min(ac + 4, board_size[1])
# 	relcol_end = relcol_start + (col_end - col_start)
#
# 	# Padd subboard to 7x7 dim
# 	sub_board = np.zeros((2, 7, 7), dtype=np.uint8)
# 	sub_board[..., relrow_start : relrow_end, relcol_start : relcol_end] = board[..., row_start : row_end, col_start : col_end]
#
# 	# Keep align stone around action in each direction
# 	captures = CAPTURE_MASK & sub_board
#
# 	# Count align stone per line
# 	captures = np.sum(captures.reshape(8, 2 * 7 * 7), axis=-1)
#
# 	# If 3 stones are align, capture is present
# 	capture_flag = captures == 3
#
# 	capture_count = np.count_nonzero(capture_flag)
# 	# if capture_count > 0:
# 	# 	# print(CAPTURE_MASK[capture_flag, 1, ...])
# 	# 	sm = np.sum(CAPTURE_MASK[capture_flag, 1, ...], axis=0, dtype=np.bool8)
# 	# 	# print(sm)
# 	# 	# sm = np.add.reduce(CAPTURE_MASK[capture_flag, 1, ...], axis=0, dtype=np.bool8)
# 	# 	board[..., 1, row_start : row_end, col_start : col_end] ^= sm[..., relrow_start : relrow_end, relcol_start : relcol_end]
#
# 	return capture_count


class Capture(AbstractRule):

	name = 'Capture'

	def __init__(self, engine: Any) -> None:
		super().__init__(engine)
		# self.CAPTURE_MASK=init_capture_mask()
		self.player_count_capture = [0, 0]

	def endturn(self, action: GomokuAction):

		c_board = ffi.cast("char *", self.engine.state.board.ctypes.data)
		y, x = action.action

		count1 = fastcore.count_captures(c_board, y, x, *self.engine.game_zone)
		self.player_count_capture[self.engine.player_idx] += count1

	def winning(self, action: GomokuAction):
		if self.player_count_capture[self.engine.player_idx] >= 5:
			raise ForceWinPlayer(reason="Five captures.")
		return False

	def create_snapshot(self):
		return {
			'player_count_capture': self.player_count_capture.copy()
		}
	
	def update_from_snapshot(self, snapshot):
		self.player_count_capture = snapshot['player_count_capture']

	def copy(self, engine: Gomoku, rule: AbstractRule):
		newrule = Capture(engine)
		newrule.player_count_capture = rule.player_count_capture.copy()
		# print("copy: ", newrule.player_count_capture, rule.player_count_capture, rule)
		return newrule
