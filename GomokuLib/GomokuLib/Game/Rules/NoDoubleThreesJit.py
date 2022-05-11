import numpy as np
import numba as nb
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore
import cffi
ffi = cffi.FFI()

cffi_utils.register_module(fastcore._rules)
is_double_threes_ctype = cffi_utils.make_function_type(fastcore._rules.lib.is_double_threes)


spec = [
	('name', nb.types.string),
	('restricting', nb.types.boolean),
	# ('FT_IDENT', nb.types.int8[:, :, :]),
	('_is_double_threes_cfunc', is_double_threes_ctype),
	('_board_ptr', nb.types.CPointer(nb.types.int8)),
	# ('_full_board_ptr', nb.types.CPointer(nb.types.int8)),
]


@jitclass(spec)
class NoDoubleThreesJit:

	def __init__(self, board: np.ndarray):
		self.name = 'NoDoubleThrees'
		self.restricting = True  # Imply existing methods get_valid() and is_valid()
		# self.FT_IDENT = init_ft_ident()
		self._is_double_threes_cfunc = fastcore._rules.lib.is_double_threes
		self._board_ptr = ffi.from_buffer(board)

	def get_valid(self, full_board: np.ndarray):
		# return njit_get_valid(board, self.FT_IDENT)
		a = np.zeros_like(full_board, dtype=np.int8)
		for r in range(19):
			for c in range(19):
				a[r, c] = self.is_valid(full_board, r, c)

	def is_valid(self, full_board: np.ndarray, ar: int, ac: int):
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

#
# @njit
# def init_ft_ident():
# 	FT_IDENT = np.zeros((4, 6, 2), dtype=np.bool8)
#
# 	FT_IDENT[0, 1, 0] = 1
# 	FT_IDENT[0, 2, 0] = 1
# 	FT_IDENT[0, 4, 0] = 1
#
# 	FT_IDENT[1, 1, 0] = 1
# 	FT_IDENT[1, 3, 0] = 1
# 	FT_IDENT[1, 4, 0] = 1
#
# 	FT_IDENT[2, 1, 0] = 1
# 	FT_IDENT[2, 2, 0] = 1
# 	FT_IDENT[2, 3, 0] = 1
#
# 	FT_IDENT[3, 2, 0] = 1
# 	FT_IDENT[3, 3, 0] = 1
# 	FT_IDENT[3, 4, 0] = 1
#
# 	return FT_IDENT
#
#
# @njit(parallel=True, nogil=True, fastmath=True)
# # @njit()
# def njit_is_valid(board, ar, ac, FT_IDENT):
#
# 	n_free_threes = 0
# 	ways = [
# 		(1, 1),
# 		(1, 0),
# 		(1, -1),
# 		(0, -1),
# 	]
# 	# print("[njit_is_valid]")
# 	for dr, dc in ways:
#
# 		r, c = ar - dr, ac - dc
# 		# print(f"\nr c dr dc: {r}, {c}, {dr}, {dc}")
#
# 		flag = 0
# 		i = 0
# 		while i < 4 and flag == 0:
# 			if dr != 0:		# WTF ???
# 				rend, rstep = r + dr * 6, dr
# 			else:
# 				rend, rstep = r, 0
#
# 			if dc != 0:
# 				cend, cstep = c + dc * 6, dc
# 			else:
# 				cend, cstep = c, 0
#
# 			start_in_bound = r >= 0 and r < 19 and c >=0 and c < 19
# 			end_in_bound = rend >= 0 and rend < 19 and cend >=0 and cend < 19
# 			if start_in_bound and end_in_bound:
#
# 				three = np.full(4, 1, dtype=np.uint8)
# 				l = 0
# 				x, y = r, c
# 				while l < 6:
#
# 					boardv = board[:, x, y]
# 					k = 0
# 					while k < 4:
# 						if np.any(boardv != FT_IDENT[k][l]):
# 							three[k] = 0
# 						k += 1
# 					x += rstep
# 					y += cstep
# 					l += 1
# 				if np.any(three == 1):
# 					flag = 1
#
# 			r -= dr
# 			c -= dc
# 			i += 1
# 			# print(f" - n_free_threes: {n_free_threes} / flag {flag}")
#
# 		n_free_threes += flag
#
# 		if n_free_threes == 2:
# 			return 0
#
# 	return 1
#
#
# @njit(parallel=True, nogil=True, fastmath=True)
# # @njit()
# def njit_get_valid(board, FT_IDENT):
# 	return np.array([
# 		[
# 			njit_is_valid(board, r, c, FT_IDENT)
# 			for c in prange(19)
# 		]
# 		for r in prange(19)
# 	], order='C')

