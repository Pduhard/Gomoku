import numba as nb
import numpy as np
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore
import cffi
ffi = cffi.FFI()

cffi_utils.register_module(fastcore._rules)
count_captures_ctype = cffi_utils.make_function_type(fastcore._rules.lib.count_captures)


class ForceWinPlayer(Exception):
    def __init__(self, reason="No reason", *args: object) -> None:
        super().__init__(args)
        self.reason = reason


spec = [
	('name', nb.types.string),
	('player_count_capture', nb.types.Any),
	('_board_ptr', nb.types.CPointer(nb.types.int8)),
	('count_captures_cfunc', count_captures_ctype),
	# ('_full_board_ptr', nb.types.CPointer(nb.types.int8)),
]


@jitclass(spec)
class CaptureJit():

	def __init__(self, board: np.ndarray) -> None:
		# self.CAPTURE_MASK=init_capture_mask()
		self.name = 'Capture'
		self.player_count_capture = [0, 0]
		self._board_ptr = ffi.from_buffer(board)
		self.count_captures_cfunc = fastcore._rules.lib.count_captures

	def endturn(self, player_idx: int, ar: int, ac: int, gz0: int, gz1: int, gz2: int, gz3: int):

		count1 = self.count_captures_cfunc(self._board_ptr, ar, ac, gz0, gz1, gz2, gz3)
		self.player_count_capture[player_idx] += count1

	def winning(self, player_idx: int, *args):
		if self.player_count_capture[player_idx] >= 5:
			raise ForceWinPlayer(reason="Five captures.")
		return False

	def get_current_player_captures(self, player_idx: int):
		return self.player_count_capture[::-1] if player_idx else self.player_count_capture

	def create_snapshot(self):
		d = dict()
		d['player_count_capture'] = self.player_count_capture.copy()
		return d

	def update_from_snapshot(self, snapshot: dict):
		self.player_count_capture = snapshot['player_count_capture']

	def copy(self, board: np.ndarray):
		newrule = CaptureJit(board)
		newrule.player_count_capture = self.player_count_capture.copy()
		return newrule




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
