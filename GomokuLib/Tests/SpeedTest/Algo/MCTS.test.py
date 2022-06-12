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
def fake_tobytes(arr: Typing.BoardDtype):
    # flat = arr.flatten()
    # i = 0
    # res = np.empty_like(flat) ('2' if arr[1, i / 19, i % 19] == 1 else '0'))

    a = []
    for i in range(19):
        for j in range(19):
            a.append('1' if arr[0, i, j] == 1 else ('2' if arr[1, i, j] == 1 else '0'))
    return ''.join(a)
    # return ''.join([
    #     ('1' if arr[0, i // 19, i % 19] == 1 else ('2' if arr[1, i // 19, i % 19] == 1 else '0'))
    #     for i in range(361)
    # ])

    return '0' * 361 + '1' * 361
    return res

@njit()
def _tobytes(n, mcts, boards):
    for i in range(n):
        mcts.fast_tobytes(boards[i])

@njit()
def _faketobytes(n, mcts, boards):
    for i in range(n):
        fake_tobytes(boards[i])

@njit()
def _expand(n, mcts, boards, actions, prunings):
    for i in range(n):
        mcts.engine.board = boards[i]
        statehash = mcts.fast_tobytes(mcts.engine.board)
        mcts.expand(statehash, actions[i], prunings[i])

@njit()
def _award(n, mcts, boards):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.award()

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
    boards = np.zeros((10000, 2, 19, 19), dtype=Typing.BoardDtype)
    old_rollingout = mcts.rollingout_turns
    mcts.rollingout_turns = 100
    _rollingout(10, mcts, boards, engine_ref)
    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        mcts.engine.update(engine_ref)
        _rollingout(r, mcts, boards, engine_ref)
        times.append(time.perf_counter())

    mcts.rollingout_turns = old_rollingout
    _log('rollingout', times, ranges)

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


def test_faketobytes(mcts, boards):

    _faketobytes(1, mcts, boards)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _faketobytes(r, mcts, boards)
        times.append(time.perf_counter())

    _log('faketobytes', times, ranges)


def test_prunning(mcts, boards):

    _prunning(1, mcts, boards)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _prunning(r, mcts, boards)
        times.append(time.perf_counter())

    _log('prunning', times, ranges)

def test_call(mcts, boards, engine_ref):

    mcts.init()
    mcts.do_n_iter(engine_ref, 100)

    times = []
    ranges = test_ranges
    nr = []
    times.append(time.perf_counter())
    for r in ranges:
        mcts.init()
        mcts.do_n_iter(engine_ref, r)

        times.append(time.perf_counter())

    _log('call', times, ranges)


def test_expand(mcts, boards):

    actions = np.random.randint(0, 2, (10000, 19, 19), dtype=Typing.ActionDtype)
    prunings = np.random.randn(10000, 19, 19) * 2.
    mcts.init()
    _expand(1, mcts, boards, actions, prunings)

    times = []
    ranges = test_ranges
    nr = []
    times.append(time.perf_counter())
    for r in ranges:
        mcts.init()
        _expand(r, mcts, boards, actions, prunings)
        times.append(time.perf_counter())

    _log('expand', times, ranges)


def test_award(mcts, boards):

    _award(1, mcts, boards)

    times = []
    ranges = test_ranges
    nr = []
    times.append(time.perf_counter())
    for r in ranges:
        _award(r, mcts, boards)
        times.append(time.perf_counter())

    _log('award', times, ranges)

if __name__ == "__main__":
    boards = np.random.randint(0, 2, (10000, 2, 19, 19), dtype=Typing.BoardDtype)
    engine = Gomoku()
    engine_ref = Gomoku()
    mcts = GomokuLib.Algo.MCTSNjit(
        engine=engine,
        iter=3000,
        pruning=True,
        rollingout_turns=5
    )
    # test_tobytes(mcts, boards)
    # test_faketobytes(mcts, boards)
    # test_prunning(mcts, boards)
    # test_get_neighbors_mask(mcts, boards)
    # test_call(mcts, boards, engine_ref)
    for i in range(10):
        test_expand(mcts, boards)
    # test_award(mcts, boards)
    # test_rollingout(mcts, boards, engine_ref)
