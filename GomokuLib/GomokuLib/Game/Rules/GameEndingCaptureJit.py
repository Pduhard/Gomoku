import numpy as np
import numba as nb
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore
import cffi
ffi = cffi.FFI()

cffi_utils.register_module(fastcore._rules)
is_winning_ctype = cffi_utils.make_function_type(fastcore._rules.lib.is_winning)


class ForceWinOpponent(Exception):
	def __init__(self, reason, *args: object) -> None:
		super().__init__(args)
		self.reason = reason or "No reason"


spec = [
	('name', nb.types.string),
	('last_capture', nb.types.Any),
	('check_ending_capture', nb.types.Any),
	('_board_ptr', nb.types.CPointer(nb.types.int8)),
	('is_winning_cfunc', is_winning_ctype),
	# ('_full_board_ptr', nb.types.CPointer(nb.types.int8)),
]


@jitclass(spec)
class GameEndingCaptureJit:

	def __init__(self, board: np.ndarray):
		self.name = 'GameEndingCapture'
		self.last_capture = [None, None]
		self.check_ending_capture = [0, 0]
		self._board_ptr = ffi.from_buffer(board)
		self.is_winning_cfunc = fastcore._rules.lib.is_winning

	def winning(self, player_idx: int, ar: int, ac: int, gz0: int, gz1: int, gz2: int, gz3: int):
		if self.check_ending_capture[player_idx ^ 1] == 0:
			return False

		win = fastcore.is_winning(self._board_ptr, 361, *self.last_capture[player_idx ^ 1], gz0, gz1, gz2, gz3)

		if win:
			raise ForceWinOpponent("GameEndingCapture")

		self.check_ending_capture[player_idx ^ 1] = 0
		return False

	def nowinning(self, player_idx: int, last_action: tuple):
		self.last_capture[player_idx] = last_action
		self.check_ending_capture[player_idx] = 1
		return True

	def create_snapshot(self):
		return {
			'last_capture': self.last_capture.copy(),
			'check_ending_capture': self.check_ending_capture.copy()
		}

	def update_from_snapshot(self, snapshot: dict):	# Remove these copy()
		self.last_capture = snapshot['last_capture'].copy()
		self.check_ending_capture = snapshot['check_ending_capture'].copy()

	def copy(self, board: np.ndarray):
		rule = GameEndingCaptureJit(board)
		rule.last_capture = self.last_capture.copy()
		rule.check_ending_capture = self.check_ending_capture.copy()
		return rule
