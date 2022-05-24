import numpy as np
import numba as nb
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore._rules as _fastcore

from GomokuLib import Typing

cffi_utils.register_module(_fastcore)
_rules = _fastcore.lib
ffi = _fastcore.ffi

is_double_threes_ctype = cffi_utils.make_function_type(_rules.is_double_threes)

@jitclass
class NoDoubleThrees:

	name:  nb.types.string
	restricting:  nb.types.boolean
	_is_double_threes_cfunc:  is_double_threes_ctype
	_board_ptr:  Typing.nbBoardFFI

	def __init__(self, board):
		self.name = 'NoDoubleThrees'
		self.restricting = True  # Imply existing methods get_valid() and is_valid()
		# self.FT_IDENT = init_ft_ident()
		self._is_double_threes_cfunc = _rules.is_double_threes
		self._board_ptr = ffi.from_buffer(board)

	def get_valid(self, full_board: np.ndarray):
		# return njit_get_valid(board, self.FT_IDENT)
		a = np.zeros_like(full_board, dtype=np.int8)
		for r in range(19):
			for c in range(19):
				a[r, c] = self.is_valid(full_board, r, c)
		return a

	def is_valid(self, full_board: np.ndarray, ar: int, ac: int):
		# return 1
		# return njit_is_valid(board, ac, ar, self.FT_IDENT)
		full_board_ptr = ffi.from_buffer(full_board)
		ret = self._is_double_threes_cfunc(self._board_ptr, full_board_ptr, ar, ac)
		return False if ret else True

	def create_snapshot(self):
		return 0

	def update_from_snapshot(self, *args):
		pass

	def update(self, *args):
		pass

	def update_board_ptr(self, board):
		self._board_ptr = ffi.from_buffer(board)