import numba as nb
import numpy as np
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore
import cffi
ffi = cffi.FFI()

cffi_utils.register_module(fastcore._rules)
is_winning_ctype = cffi_utils.make_function_type(fastcore._rules.lib.is_winning)


spec = [
    ('name', nb.types.string),
    ('restricting', nb.types.boolean),
    ('_winning_cfunc', is_winning_ctype),
    ('_board_ptr', nb.types.CPointer(nb.types.int8)),
    # ('_full_board_ptr', nb.types.CPointer(nb.types.int8)),
]

@jitclass(spec)
class BasicRuleJit:

    def __init__(self, board: np.ndarray):
        self.name = 'BasicRule'
        self.restricting = True  # Imply existing methods get_valid() and is_valid()
        self._winning_cfunc = fastcore._rules.lib.is_winning
        self._board_ptr = ffi.from_buffer(board)

    def get_valid(self, full_board: np.ndarray):
        return full_board ^ 1

    def is_valid(self, full_board: np.ndarray, ar: int, ac: int):
        return full_board[ar, ac] == 0

    def winning(self, player_idx: int, ar: int, ac: int, gz0: int, gz1: int, gz2: int, gz3: int):
        return self._winning_cfunc(self._board_ptr, 0, ar, ac, gz0, gz1, gz2, gz3)

    def create_snapshot(self):
        return 0

    def update_from_snapshot(self, *args):
        pass

    def update(self, *args):
        pass

    def update_board_ptr(self, board):
        self._board_ptr = ffi.from_buffer(board)