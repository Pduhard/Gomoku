import numba
from numba import typed
from numba.core import types
from numba.experimental import jitclass

from cffi import FFI
import fastcore
from fastcore._rules import ffi, lib
from numba.core.typing import cffi_utils

cffi_utils.register_module(fastcore._rules)

# @numba.njit("boolean(pyobject, char[:], int8, int8, int8, int8, int8, int8)")
# def _C_winning(board, ar, ac, gz0, gz1, gz2, gz3):
#     c_board = ffi.cast("char *", board.ctypes.data)
#     return fastcore.is_winning(c_board, ar, ac, gz0, gz1, gz2, gz3)


spec = [
    ('name', types.string),
    ('restricting', types.boolean),
]

fn = lib.is_winning
@jitclass(spec)
class BasicRuleJit:

    def __init__(self):
        self.name = 'BasicRule'
        self.restricting = True  # Imply existing methods get_valid() and is_valid()

    def get_valid(self, full_board):
        return full_board ^ 1

    def is_valid(self, full_board, ar, ac):
        return full_board[ar, ac] == 0

    def winning(self, board, ar, ac, gz0, gz1, gz2, gz3):

        return fn(board, ar, ac, gz0, gz1, gz2, gz3)

        ways = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1)
        ]

        # 4 direction
        for rway, cway in ways:

            # Slide 4 times
            r1, c1, count1, i = ar, ac, 4, 0
            while (i < 4 and count1 == 4):

                r1 += rway
                c1 += cway
                if (r1 < gz0 or r1 >= gz1 or c1 < gz2 or c1 >= gz3 or board[0, r1, c1] == 0):
                    count1 = i
                i += 1

            r2, c2, count2, i = ar, ac, 4, 0
            while (i < 4 and count2 == 4):
                r2 -= rway
                c2 -= cway
                if (r2 < gz0 or r2 >= gz1 or c2 < gz2 or c2 >= gz3 or board[0, r2, c2] == 0):
                    count2 = i
                i += 1

            if count1 + count2 >= 4:
                return True

        return False

    def create_snapshot(self):
        return typed.Dict.empty(types.string, types.string)

    def update_from_snapshot(self):
        pass

    def copy(self):
        return BasicRuleJit()
