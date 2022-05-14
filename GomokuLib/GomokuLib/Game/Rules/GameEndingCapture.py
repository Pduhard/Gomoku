import numpy as np
import numba as nb
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore._rules as _fastcore

cffi_utils.register_module(_fastcore)
_rules = _fastcore.lib
ffi = _fastcore.ffi

is_winning_ctype = cffi_utils.make_function_type(_rules.is_winning)

spec = [
	('name', nb.types.string),
	('stats', nb.types.int8[:, :]),
	('_board_ptr', nb.types.CPointer(nb.types.int8)),
	('is_winning_cfunc', is_winning_ctype),
]


@jitclass(spec)
class GameEndingCapture:

	def __init__(self, board: np.ndarray):
		self.name = 'GameEndingCapture'
		self.stats = np.zeros((3, 2), dtype=np.int8)
		self._board_ptr = ffi.from_buffer(board)
		self.is_winning_cfunc = _rules.is_winning

	def winning(self, player_idx: int, ar: int, ac: int, gz0: int, gz1: int, gz2: int, gz3: int):
		if self.stats[2][player_idx ^ 1] == 0:
			return 0
		ar, ac = self.stats[player_idx ^ 1]

		win = self.is_winning_cfunc(self._board_ptr, 361, ar, ac, gz0, gz1, gz2, gz3)
		if win:
			return 3

		self.stats[2][player_idx ^ 1] = 0
		return 0

	def endturn(self, player_idx: int, ar: int, ac: int, *args):
		self.stats[2][player_idx] = 1
		self.stats[player_idx] = (ar, ac)

	def create_snapshot(self):
		return self.stats

	def update_from_snapshot(self, stats: np.ndarray):	# Remove these copy()
		self.stats[...] = stats

	def update(self, rule):
		self.stats[...] = rule.stats

	def update_board_ptr(self, board):
		self._board_ptr = ffi.from_buffer(board)