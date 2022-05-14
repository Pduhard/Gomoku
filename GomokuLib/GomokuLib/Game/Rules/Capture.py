import numba as nb
import numpy as np
from numba import njit
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore._rules as _fastcore

cffi_utils.register_module(_fastcore)
_rules = _fastcore.lib
ffi = _fastcore.ffi

count_captures_ctype = cffi_utils.make_function_type(_rules.count_captures)

class ForceWinPlayer(Exception):
    def __init__(self, reason="No reason", *args: object) -> None:
        super().__init__(args)
        self.reason = reason

spec = [
	('name', nb.types.string),
	('player_count_capture', nb.types.int8[:]),
	('_board_ptr', nb.types.CPointer(nb.types.int8)),
	('count_captures_cfunc', count_captures_ctype),
]

@jitclass(spec)
class Capture():

	def __init__(self, board: np.ndarray) -> None:
		self.name = 'Capture'
		self.player_count_capture = np.zeros(2, dtype=np.int8)
		self._board_ptr = ffi.from_buffer(board)
		self.count_captures_cfunc = _rules.count_captures
	
	def reset_values(self):
		self.player_count_capture = np.zeros(2, dtype=np.int8)

	def endturn(self, player_idx: int, ar: int, ac: int, gz0: int, gz1: int, gz2: int, gz3: int):

		count1 = self.count_captures_cfunc(self._board_ptr, ar, ac, gz0, gz1, gz2, gz3)
		self.player_count_capture[player_idx] += count1

	def winning(self, player_idx: int, *args):
		if self.player_count_capture[player_idx] >= 5:
			return 2
		return 0

	def get_current_player_captures(self, player_idx: int):
		return self.player_count_capture[::-1] if player_idx else self.player_count_capture

	def create_snapshot(self):
		return self.player_count_capture

	def update_from_snapshot(self, player_count_capture):
		self.player_count_capture[0] = player_count_capture[0]
		self.player_count_capture[1] = player_count_capture[1]

	def update(self, rule):
		self.player_count_capture[:] = rule.player_count_capture

	def update_board_ptr(self, board):
		self._board_ptr = ffi.from_buffer(board)