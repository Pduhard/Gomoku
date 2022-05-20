from time import perf_counter
from numba import njit
import numpy as np
import time
from cffi import FFI
ffi = FFI()
from GomokuLib.Game.Rules import Capture
from GomokuLib import Typing

test_ranges = [10, 1000, 1000000]

# @njit()
# def _get_valid(n, full_board, capture):
#     # fb = ffi.from_buffer(full_board)
#     ret = np.zeros_like(full_board, dtype=Typing.BoardDtype)
#     for i in range(n):
#         ret += capture.get_valid(full_board)
#     return ret


@njit()
def _endturn(n, capture):
    for i in range(n):
        capture.endturn(
            i % 2, i / 19, i % 19, 0, 0, 18, 18
        )

@njit()
def _winning(n, capture):
    for i in range(n):
        capture.winning(
            i % 2
        )

def _log(fname, times, ranges):
    print('######################')
    print('Capture', fname, ': ')
    print('######################')
    avgt = 0
    avgc = 0
    for t, t1, r in zip(times[:-1], times[1:], ranges):
        avgt += t1 - t
        avgc += r
        print(r, f'times in {(t1 - t) * 1000:.4f} ms')
    avg = avgt / avgc
    for unit in ["s", "ms", "us", "ns"]:
        if avg > 1:
            print(f'avg call time: {avg} {unit}')
            return
        avg *= 1000
    avg /= 1000
    print(f'avg call time: {avg / 1000} ns')

    
    

# def test_get_valid():
#     capture = Capture(np.zeros((2, 19, 19), dtype=Typing.BoardDtype))
#     full_board = np.random.randint(0, 2, (19, 19), dtype=Typing.BoardDtype)
#     _get_valid(1, full_board, capture)

#     times = []
#     ranges = test_ranges

#     times.append(time.perf_counter())
#     for r in ranges:
#         _get_valid(r, full_board, capture)
#         times.append(time.perf_counter())

#     _log('get_valid', times, ranges)

def test_endturn():
    capture = Capture(np.random.randint(0, 2, (2, 19, 19), dtype=Typing.BoardDtype))
    _endturn(1, capture)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _endturn(r, capture)
        times.append(time.perf_counter())

    _log('endturn', times, ranges)


def test_winning():
    capture = Capture(np.random.randint(0, 2, (2, 19, 19), dtype=Typing.BoardDtype))
    _winning(1, capture)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _winning(r, capture)
        times.append(time.perf_counter())

    _log('winning', times, ranges)

if __name__ == "__main__":
    # test_get_valid()
    # test_is_valid()
    test_winning()
    test_endturn()

