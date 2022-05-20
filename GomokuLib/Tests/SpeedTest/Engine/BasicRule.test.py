from time import perf_counter
from numba import njit
import numpy as np
import time
from cffi import FFI
ffi = FFI()
from GomokuLib.Game.Rules import BasicRule
from GomokuLib import Typing

test_ranges = [10, 1000, 1000000]

@njit()
def _get_valid(n, full_board, basic_rule):
    # fb = ffi.from_buffer(full_board)
    ret = np.zeros_like(full_board, dtype=Typing.BoardDtype)
    for i in range(n):
        ret += basic_rule.get_valid(full_board)
    return ret


@njit()
def _is_valid(n, full_board, basic_rule):
    ret = dtype=Typing.BoardDtype(0)
    for i in range(n):
        ret += basic_rule.is_valid(
            full_board, (i ^ 78) % 19, (i ^ 23) % 19
        )
    return ret

@njit()
def _winning(n, basic_rule):
    ret = np.zeros((2, 19, 19), dtype=Typing.BoardDtype)
    for i in range(n):
        ret ^= basic_rule.winning(
            0, i / 19, i % 19, 0, 0, 18, 18
        )
    return ret

@njit()
def _update_board_ptr(n, basic_rule, boards):
    for i in range(n):
        basic_rule.update_board_ptr(
            np.copy(boards[i])
        )


def _log(fname, times, ranges):
    print('######################')
    print('BasicRule', fname, ': ')
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

def test_get_valid():
    basic_rule = BasicRule(np.zeros((2, 19, 19), dtype=Typing.BoardDtype))
    full_board = np.random.randint(0, 2, (19, 19), dtype=Typing.BoardDtype)
    _get_valid(1, full_board, basic_rule)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _get_valid(r, full_board, basic_rule)
        times.append(time.perf_counter())

    _log('get_valid', times, ranges)

def test_is_valid():
    basic_rule = BasicRule(np.zeros((2, 19, 19), dtype=Typing.BoardDtype))
    full_board = np.random.randint(0, 2, (19, 19), dtype=Typing.BoardDtype)
    _is_valid(1, full_board, basic_rule)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _is_valid(r, full_board, basic_rule)
        times.append(time.perf_counter())

    _log('is_valid', times, ranges)


def test_winning():
    basic_rule = BasicRule(np.random.randint(0, 2, (2, 19, 19), dtype=Typing.BoardDtype))
    _winning(1, basic_rule)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _winning(r, basic_rule)
        times.append(time.perf_counter())

    _log('winning', times, ranges)

def test_update_board_ptr():
    ## Weird test can't determine what's happening
    # because of boards generation

    basic_rule = BasicRule(np.random.randint(0, 2, (2, 19, 19), dtype=Typing.BoardDtype))
    _update_board_ptr(1, basic_rule, np.random.randint(0, 2, (2, 2, 19, 19), dtype=Typing.BoardDtype))
    times = []
    ranges = test_ranges

    for r in ranges:
        boards = np.random.randint(0, 2, (r, 2, 19, 19), dtype=Typing.BoardDtype)
        times.append(time.perf_counter())
        _update_board_ptr(r, basic_rule, boards)
        times.append(time.perf_counter())

    _log('updateboardptr', times, ranges)

if __name__ == "__main__":
    # test_get_valid()
    test_is_valid()
    test_winning()
    test_update_board_ptr()

