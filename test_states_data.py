from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from numba import njit
import numpy as np
from fastcore._rules import ffi, lib


@njit()
def method_1(engine, arr):
    engine.board = arr.copy()
    cpt = ffi.from_buffer(engine.board)
    lib.is_winning(cpt, 0, 0, 0, 0, 0, 0, 0)

@njit()
def method_2(engine, arr, cpt):
    engine.board = arr.copy()
    lib.is_winning(cpt, 0, 0, 0, 0, 0, 0, 0)

@njit()
def method_3(engine, arr):
    engine.board = arr.copy()
    cpt = ffi.from_buffer(engine.board)
    lib.is_winning(cpt, 0, 0, 0, 0, 0, 0, 0)

engine = Gomoku()
arr = np.ones_like(engine.board)

method_1(engine, arr)
cpt = ffi.from_buffer(engine.board)
method_2(engine, arr, cpt)
method_3(engine, arr)

from time import perf_counter

s = perf_counter()
cb1 = ffi.from_buffer(engine.board)
for i in range(100): method_1(engine, arr)
s1 = perf_counter()
cb2 = ffi.from_buffer(engine.board)
for i in range(100):
    cpt = ffi.from_buffer(engine.board)
    method_2(engine, arr, cpt)
s2 = perf_counter()
cb3 = ffi.from_buffer(engine.board)
cpt = ffi.from_buffer(engine.board)
for i in range(100): method_3(engine, arr)
s3 = perf_counter()

print('1', (s1 - s) * 1000, ' ms')
print('2', (s2 - s1) * 1000, ' ms')
print('3', (s3 - s2) * 1000, ' ms')
breakpoint()