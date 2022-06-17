import numba as nb
import numpy as np
from numba.core.typing import cffi_utils
from numba.experimental import jitclass

import fastcore._rules as _fastcore

from GomokuLib import Typing

cffi_utils.register_module(_fastcore)
_rules = _fastcore.lib
ffi = _fastcore.ffi

is_winning_ctype = cffi_utils.make_function_type(_rules.is_winning)

@jitclass
class BasicRule:

    name: nb.types.string
    restricting: nb.types.boolean
    _winning_cfunc: is_winning_ctype
    _board_ptr: Typing.nbBoardFFI

    def __init__(self, board):
        self.name = 'BasicRule'
        self.restricting = True  # Imply existing methods get_valid() and is_valid()
        self._winning_cfunc = _rules.is_winning
        self._board_ptr = ffi.from_buffer(board)

    def get_valid(self, full_board: np.ndarray):
        ret = np.ones_like(full_board)
        return ret ^ full_board

    def is_valid(self, board: np.ndarray, ar: int, ac: int):
        return board[0, ar, ac] + board[1, ar, ac] == 0

    def winning(self, player_idx: int, ar: int, ac: int, gz0: int, gz1: int, gz2: int, gz3: int):
        return self._winning_cfunc(self._board_ptr, player_idx, ar, ac, gz0, gz1, gz2, gz3)

    def create_snapshot(self):
        return 0

    def update_from_snapshot(self, *args):
        pass

    def update(self, *args):
        pass

    def update_board_ptr(self, board):
        self._board_ptr = ffi.from_buffer(board)
