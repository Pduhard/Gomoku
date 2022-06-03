from time import perf_counter
from numba import njit
import numpy as np
import time
from cffi import FFI
ffi = FFI()
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib import Typing
import GomokuLib

test_ranges = [10, 1000, 10000]

@njit()
def _get_actions(n, gomoku):
    # fb = ffi.from_buffer(full_board)
    ret = np.zeros((19, 19), dtype=Typing.BoardDtype)
    for i in range(n):
        ret += gomoku.get_actions()
    return ret


@njit()
def _get_neighbors_mask(n, mcts, boards):
    ret = 0
    for i in range(n):
        # this is a full board actually
        mcts.get_neighbors_mask(boards[i][0])
    return ret

@njit()
def _prunning(n, mcts, boards):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.pruning()        

@njit()
def _rollingout(n, mcts, boards, engine_ref):
    for i in range(n):
        mcts.engine.update(engine_ref)
        mcts.engine.board = boards[i]
        mcts.rollingout()

@njit()
def _tobytes(n, mcts, boards):
    for i in range(n):
        mcts.tobytes(boards[i])

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


def test_rollingout(mcts, boards, engine_ref):
    mcts.rollingout_turns = 100
    _rollingout(10, mcts, boards, engine_ref)
    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        mcts.engine.update(engine_ref)
        _rollingout(r, mcts, boards, engine_ref)
        times.append(time.perf_counter())

    _log('get_neighbors_mask', times, ranges)

def test_get_neighbors_mask(mcts, boards):

    _get_neighbors_mask(1, mcts, boards)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _get_neighbors_mask(r, mcts, boards)
        times.append(time.perf_counter())

    _log('get_neighbors_mask', times, ranges)

def test_tobytes(mcts, boards):

    _tobytes(1, mcts, boards)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _tobytes(r, mcts, boards)
        times.append(time.perf_counter())

    _log('tobytes', times, ranges)


def test_prunning(mcts, boards):

    _prunning(1, mcts, boards)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _prunning(r, mcts, boards)
        times.append(time.perf_counter())

    _log('prunning', times, ranges)

if __name__ == "__main__":
    boards = np.random.randint(0, 2, (10000, 2, 19, 19), dtype=Typing.BoardDtype)
    engine = Gomoku()
    engine_ref = Gomoku()
    mcts = GomokuLib.Algo.MCTSNjit(
        engine=engine,
        iter=3000,
        pruning=True,
        rollingout_turns=10
    )
    test_tobytes(mcts, boards)
    test_prunning(mcts, boards)
    test_get_neighbors_mask(mcts, boards)
    test_rollingout(mcts, boards, engine_ref)
