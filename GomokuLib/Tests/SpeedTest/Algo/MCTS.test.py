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
def _fast_tobytes(n, mcts, boards):
    for i in range(n):
        mcts.fast_tobytes(boards[i])

@njit()
def _expand(n, mcts, boards, actions, prunings):
    for i in range(n):
        mcts.engine.board = boards[i]
        statehash = mcts.fast_tobytes(mcts.engine.board)
        mcts.expand(statehash, actions[i], prunings[i])

@njit()
def _backprop_memory(n, mcts, boards, best_actions, rewards, statehashes):
    for i in range(n):
        mcts.backprop_memory(best_actions[i], rewards[i], statehashes[i])

@njit()
def _get_policy(n, mcts, state_data):
    for i in range(n):
        mcts.get_policy(state_data[i])


@njit()
def _get_best_policy_actions(n, mcts, policies, actions):
    for i in range(n):
        mcts.get_best_policy_actions(policies[i], actions[i])

@njit()
def _lazy_selection(n, mcts, policies, actions):
    for i in range(n):
        mcts.lazy_selection(policies[i], actions[i])

@njit()
def _award(n, mcts, boards):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.award()
@njit()
def _heuristic(n, mcts, boards):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.heuristic()

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

    _fast_tobytes(1, mcts, boards)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _fast_tobytes(r, mcts, boards)
        times.append(time.perf_counter())

    _log('fast_tobytes', times, ranges)

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
    times.append(time.perf_counter())
    for r in ranges:
        _award(r, mcts, boards)
        times.append(time.perf_counter())

    _log('award', times, ranges)

def test_get_policy(mcts):
    state_datas = np.empty((10000), dtype=Typing.StateDataDtype)

    _get_policy(1, mcts, state_datas)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _get_policy(r, mcts, state_datas)
        times.append(time.perf_counter())

    _log('get_policy', times, ranges)

def test_get_best_policy_actions(mcts):
    policies = np.random.randn(10000, 19, 19).astype(np.float64)
    actions = np.random.randint(0, 2, (10000, 19, 19), dtype=np.int8)

    _get_best_policy_actions(1, mcts, policies, actions)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _get_best_policy_actions(r, mcts, policies, actions)
        times.append(time.perf_counter())

    _log('get_best_policy_actions', times, ranges)

def test_lazy_selection(mcts):
    policies = np.around(np.random.randn(10000, 19, 19).astype(np.float64), decimals=2)
    actions = np.random.randint(0, 2, (10000, 19, 19), dtype=np.int8)

    _lazy_selection(1, mcts, policies, actions)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _lazy_selection(r, mcts, policies, actions)
        times.append(time.perf_counter())

    _log('lazy_selection', times, ranges)

def test_heuristic(mcts, boards):

    _heuristic(1, mcts, boards)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _heuristic(r, mcts, boards)
        times.append(time.perf_counter())

    _log('heuristic', times, ranges)

def test_backprop_memory(mcts, boards):

    best_actions = np.random.randint(0, 18, (10000, 2), dtype=Typing.ActionDtype)
    rewards = np.random.randn(10000) * 2.
    statehashes = [mcts.fast_tobytes(b) for b in boards]
    _backprop_memory(1, mcts, boards, best_actions, rewards, statehashes)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _backprop_memory(r, mcts, boards, best_actions, rewards, statehashes)
        times.append(time.perf_counter())

    _log('backprop_memory', times, ranges)

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
    engine.game_zone = np.array((0, 0, 18, 18), dtype=Typing.GameZoneDtype)
    test_tobytes(mcts, boards)
    test_prunning(mcts, boards)
    test_get_neighbors_mask(mcts, boards)
    test_call(mcts, boards, engine_ref)
    test_expand(mcts, boards)
    test_backprop_memory(mcts, boards) # do it after expand
    test_award(mcts, boards)
    test_rollingout(mcts, boards, engine_ref)
    test_get_policy(mcts)
    test_get_best_policy_actions(mcts)
    test_lazy_selection(mcts)
    test_heuristic(mcts, boards)
    # test_backpropagation() ??