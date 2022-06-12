import numba as nb
import numpy as np
from numba import njit
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore._rules as _fastcore

from GomokuLib import Typing

cffi_utils.register_module(_fastcore)
_rules = _fastcore.lib
ffi = _fastcore.ffi

count_captures_ctype = cffi_utils.make_function_type(_rules.count_captures)


@jitclass
class Capture():
	name: nb.types.string
	player_count_capture: Typing.nbTuple
	_board_ptr: Typing.nbBoardFFI
	captured_buffer: Typing.nbCapturedBuf 
	_captured_ptr: Typing.nbCapturedBufFFI
	count_captures_cfunc: count_captures_ctype
	capture_count: Typing.mcts_int_nb_dtype

	def __init__(self, board) -> None:
		self.name = 'Capture'
		self.player_count_capture = np.zeros(2, dtype=Typing.TupleDtype)
		self._board_ptr = ffi.from_buffer(board)
		self.count_captures_cfunc = _rules.count_captures
		self.captured_buffer = np.zeros((16, 2, 2), dtype=np.int32)
		self._captured_ptr = ffi.from_buffer(self.captured_buffer)

	def endturn(self, player_idx: int, ar: int, ac: int, gz0: int, gz1: int, gz2: int, gz3: int):
		self.capture_count = self.count_captures_cfunc(self._board_ptr, ar, ac, gz0, gz1, gz2, gz3, player_idx, self._captured_ptr)
		self.player_count_capture[player_idx] += self.capture_count

	def winning(self, player_idx: int, *args):
		if self.player_count_capture[player_idx] >= 5:
			return 2
		return 0

	def get_captures(self):
		return self.player_count_capture

	def create_snapshot(self):
		return self.player_count_capture

	def update_from_snapshot(self, player_count_capture):
		self.player_count_capture = np.copy(player_count_capture)

	def update(self, rule):
		self.player_count_capture = np.copy(rule.player_count_capture)

	def update_board_ptr(self, board):
		self._board_ptr = ffi.from_buffer(board)