import numba
from numba import typed
from numba.core import types
from numba.experimental import jitclass
import cffi
ffi = cffi.FFI()

# from cffi import FFI
# import fastcore
import fastcore._rules as c_module
# from fastcore._rules import ffi, lib as md
from numba.core.typing import cffi_utils

import ctypes

cffi_utils.register_module(c_module)
is_winning_ctype = cffi_utils.make_function_type(c_module.lib.is_winning)
is_winning = c_module.lib.is_winning

spec = [
    ('name', types.string),
    ('restricting', types.boolean),
    ('_winning_cfunc', is_winning_ctype),
]

@jitclass(spec)
class BasicRuleJit:

    def __init__(self):
        self.name = 'BasicRule'
        self.restricting = True  # Imply existing methods get_valid() and is_valid()
        self._winning_cfunc = c_module.lib.is_winning

    def get_valid(self, full_board):
        return full_board ^ 1

    def is_valid(self, full_board, ar, ac):
        return full_board[ar, ac] == 0

    def winning(self, board, ar, ac, gz0, gz1, gz2, gz3):

        ptr = ffi.from_buffer(board)
        for i in range(1000):
            for a in range(19):
                for b in range(19):
                    self._winning_cfunc(ptr, ar, ac, gz0, gz1, gz2, gz3)
        return 0

    def create_snapshot(self):
        return typed.Dict.empty(types.string, types.string)

    def update_from_snapshot(self):
        pass

    def copy(self):
        return BasicRuleJit()
