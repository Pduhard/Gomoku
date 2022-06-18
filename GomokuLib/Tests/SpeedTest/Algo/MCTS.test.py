from time import perf_counter
from numba import njit
import numpy as np
import time
from cffi import FFI
ffi = FFI()
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib import Typing
import GomokuLib

from GomokuLib.Algo.hpruning import _get_neighbors_mask as mcts_get_neighbors_mask
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
        mcts_get_neighbors_mask(boards[i][0])
    return ret

@njit()
def _prunning(n, mcts, boards, game_zones):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.engine.game_zone = game_zones[i]
        mcts.new_state_pruning()        

@njit()
def _fast_tobytes(n, mcts, boards):
    for i in range(n):
        mcts.fast_tobytes(boards[i])

@njit()
def _expand(n, mcts, boards, actions, rewards, prunings, game_zones, statehashes):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.engine.game_zone = game_zones[i]
        mcts.current_statehash = statehashes[i]
        mcts.expand(actions[i], rewards[i], prunings[i])

@njit()
def _backprop_memory(n, mcts, best_actions, rewards, statehashes):
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
def _lazy_selection(n, mcts, policies, actions, engine_ref):
    for i in range(n):
        mcts.engine.update(engine_ref)
        mcts.lazy_selection(policies[i], actions[i])

@njit()
def _award(n, mcts, boards, best_actions, statehashes):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.award(statehashes[i], best_actions[i], best_actions[i - 1])
@njit()
def _heuristic(n, mcts, boards, game_zones):
    for i in range(n):
        mcts.engine.board = boards[i]
        mcts.engine.game_zone = game_zones[i]
        mcts.heuristic()

def _log(fname, times, ranges):
    print('######################')
    print('MCTS', fname, ': ')
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

def test_prunning(mcts, boards, game_zones):

    _prunning(1, mcts, boards, game_zones)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _prunning(r, mcts, boards, game_zones)
        times.append(time.perf_counter())

    _log('prunning', times, ranges)

def test_call(mcts, boards, engine_ref):

    mcts.init()
    mcts.do_your_fck_work(engine_ref, 100, 0)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        mcts.init()
        mcts.do_your_fck_work(engine_ref, r, 0)

        times.append(time.perf_counter())

    _log('call', times, ranges)


def test_expand(mcts, boards, actions, rewards, prunings, game_zones):

    statehashes = [mcts.fast_tobytes(b) for b in boards]
    mcts.init()
    _expand(1, mcts, boards, actions, rewards, prunings, game_zones, statehashes)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        mcts.init()
        _expand(r, mcts, boards, actions, rewards, prunings, game_zones, statehashes)
        times.append(time.perf_counter())

    _log('expand', times, [10, 10, 1000, 1000, 10000, 10000])


def test_award(mcts, boards, actions):
    statehashes = [mcts.fast_tobytes(b) for b in boards]

    _award(1, mcts, boards, actions, statehashes)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _award(r, mcts, boards, actions, statehashes)
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

def test_get_best_policy_actions(mcts, policies, actions):

    _get_best_policy_actions(1, mcts, policies, actions)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _get_best_policy_actions(r, mcts, policies, actions)
        times.append(time.perf_counter())

    _log('get_best_policy_actions', times, ranges)

def test_lazy_selection(mcts, policies, actions, engine_ref):

    _lazy_selection(1, mcts, policies, actions, engine_ref)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _lazy_selection(r, mcts, policies, actions, engine_ref)
        times.append(time.perf_counter())

    _log('lazy_selection', times, ranges)

def test_heuristic(mcts, boards, game_zones):

    _heuristic(1, mcts, boards, game_zones)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _heuristic(r, mcts, boards, game_zones)
        times.append(time.perf_counter())

    _log('heuristic', times, ranges)

def test_backprop_memory(mcts, best_actions, rewards):

    statehashes = [mcts.fast_tobytes(b) for b in boards]
    _backprop_memory(1, mcts, best_actions, rewards, statehashes)

    times = []
    ranges = test_ranges
    times.append(time.perf_counter())
    for r in ranges:
        _backprop_memory(r, mcts, best_actions, rewards, statehashes)
        times.append(time.perf_counter())

    _log('backprop_memory', times, ranges)

if __name__ == "__main__":
    boards = np.random.randint(0, 2, (10000, 2, 19, 19), dtype=Typing.BoardDtype)
    # my_h_graph = init_my_heuristic_graph()
    # opp_h_graph = init_opp_heuristic_graph()
    # my_cap_graph = init_my_captures_graph()
    # opp_cap_graph = init_opp_captures_graph()
    # while True:
    #     board = np.random.randint(0, 10, size=(2, 19, 19), dtype=Typing.BoardDtype)
    #     board = _keep_uppers(board.astype(Typing.PruningDtype), Typing.PruningDtype(9)).astype(Typing.BoardDtype)
    #     print(board)
    #     pruning_arr = njit_create_hpruning(board, 0, 0, 18, 18, 0, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph)
        
    #     for depth in range(6):
    #         pruning = njit_dynamic_hpruning(pruning_arr, depth)
    #         print(f"pruning depth {depth}:\n{pruning}\n\n")
        
    #     breakpoint()
    best_actions = np.random.randint(0, 18, (10000, 2), dtype=Typing.ActionDtype)
    policies = np.around(np.random.randn(10000, 19, 19), decimals=2).astype(np.float64)
    actions = np.random.randint(0, 2, (10000, 19, 19), dtype=Typing.ActionDtype)
    amafs = np.random.randn(10000, 2, 2, 19, 19)
    rewards = np.random.randn(10000)
    prunings = np.random.randn(10000, 3, 19, 19) * 2.
    engine = Gomoku()
    engine_ref = Gomoku()
    game_zones = np.empty((10000, 4), dtype=Typing.GameZoneDtype)
    gzoffset = np.random.randint(0, 9, (20000, 2), dtype=Typing.ActionDtype)
    game_zones[:, :2] = gzoffset[:10000]
    game_zones[:, 2:] = gzoffset[:10000] + gzoffset[10000:]
    engine.game_zone = np.array((0, 0, 18, 18), dtype=Typing.GameZoneDtype)
    mcts = GomokuLib.Algo.MCTSNjit(
        engine=engine,
        iter=3000,
    )
    # test_tobytes(mcts, boards)
    # test_prunning(mcts, boards, game_zones)
    # test_get_neighbors_mask(mcts, boards)
    # test_call(mcts, boards, engine_ref)

    for i in range(10):
        boards = np.random.randint(0, 2, (10000, 2, 19, 19), dtype=Typing.BoardDtype)
        rewards = np.random.randn(10000)
        prunings = np.random.randn(10000, 3, 19, 19) * 2.
        actions = np.random.randint(0, 2, (10000, 19, 19), dtype=Typing.ActionDtype)
        game_zones = np.empty((10000, 4), dtype=Typing.GameZoneDtype)
        gzoffset = np.random.randint(0, 9, (20000, 2), dtype=Typing.ActionDtype)
        game_zones[:, :2] = gzoffset[:10000]
        game_zones[:, 2:] = gzoffset[:10000] + gzoffset[10000:]
        test_expand(mcts, boards, actions, rewards, prunings, game_zones)
    # test_backprop_memory(mcts, best_actions, rewards) # do it after expand
    # test_award(mcts, boards, best_actions)
    # test_get_policy(mcts)
    # test_get_best_policy_actions(mcts, policies, actions)
    # test_lazy_selection(mcts, policies, actions, engine_ref)
    # test_heuristic(mcts, boards, game_zones)
    # test_backpropagation() ??