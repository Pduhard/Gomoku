from time import perf_counter
from numba import njit
import numpy as np
import time
from cffi import FFI
ffi = FFI()
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib import Typing

test_ranges = [10, 1000, 10000, 1000000]

@njit()
def _get_actions(n, gomoku):
    # fb = ffi.from_buffer(full_board)
    ret = np.zeros((19, 19), dtype=Typing.BoardDtype)
    for i in range(n):
        ret += gomoku.get_actions()
    return ret


@njit()
def _is_valid_action(n, gomoku, random_actions):
    ret = 0
    for i in range(n):
        ret ^= gomoku.is_valid_action(
            random_actions[i]
        )
    return ret

@njit()
def _update(n, gomoku1, gomoku2):
    for i in range(n):
        cd = i & 1
        g1 = gomoku1 if cd else gomoku2
        g2 = gomoku2 if cd else gomoku1
        g1.update(g2)

# @njit()
# def _winning(n, gomoku):
#     ret = np.zeros((2, 19, 19), dtype=Typing.BoardDtype)
#     for i in range(n):
#         ret ^= gomoku.winning(
#             0, i / 19, i % 19, 0, 0, 18, 18
#         )
#     return ret

def _log(fname, times, ranges):
    print('######################')
    print('Gomoku', fname, ': ')
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

def test_get_actions():
    gomoku = Gomoku(np.zeros((2, 19, 19), dtype=Typing.BoardDtype))
    _get_actions(1, gomoku)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _get_actions(r, gomoku)
        times.append(time.perf_counter())

    _log('get_actions', times, ranges)


def test_is_valid_action():
    gomoku = Gomoku()
    gomoku.board = np.random.randint(0, 2, (2, 19, 19), dtype=Typing.BoardDtype)
    gomoku.update_board_ptr()
    _is_valid_action(1, gomoku, np.zeros((2, 2), dtype=Typing.BoardDtype))

    times = []
    ranges = test_ranges

    random_actions = [
        np.random.randint(0, 19, (r, 2), dtype=Typing.ActionDtype)
        for r in ranges
    ]
    times.append(time.perf_counter())
    for r, random_action in zip(ranges, random_actions):
        _is_valid_action(r, gomoku, random_action)
        times.append(time.perf_counter())

    _log('is_valid_action', times, ranges)

def test_update():
    gomoku1 = Gomoku()
    gomoku1.board = np.random.randint(0, 2, (2, 19, 19), dtype=Typing.BoardDtype)
    gomoku1.update_board_ptr()

    gomoku2 = Gomoku()
    gomoku2.board = np.random.randint(0, 2, (2, 19, 19), dtype=Typing.BoardDtype)
    gomoku2.update_board_ptr()

    _update(1, gomoku1, gomoku2)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _update(r, gomoku1, gomoku2)
        times.append(time.perf_counter())

    _log('update', times, ranges)

if __name__ == "__main__":
    test_get_actions()
    # test_is_valid()
    test_is_valid_action()
    test_update()

