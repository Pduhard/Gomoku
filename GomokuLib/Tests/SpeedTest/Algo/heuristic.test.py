import time
from GomokuLib import Typing
from GomokuLib.Algo.aligns_graphs import (
    init_my_heuristic_graph,
    init_opp_heuristic_graph,
    init_my_captures_graph,
    init_opp_captures_graph
)

from GomokuLib.Algo.heuristic import (
    _find_align_reward as heuristic_find_align_award,
    _compute_capture_coef,
    njit_heuristic,
    old_njit_heuristic,
)

import numpy as np
from numba import njit

test_ranges = [10, 1000, 10000]

@njit()
def _find_align_reward(n, boards, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, dirs, p):
    for i in range(n):
        heuristic_find_align_award(boards[i], my_h_graph, (i * 9) % 18, (i * 7) % 18, i % 2, p, dirs)
        heuristic_find_align_award(boards[i], opp_h_graph, (i * 9) % 18, (i * 7) % 18, i % 2, p, dirs)
        heuristic_find_align_award(boards[i], my_cap_graph, (i * 9) % 18, (i * 7) % 18, i % 2, p, dirs)
        heuristic_find_align_award(boards[i], opp_cap_graph, (i * 9) % 18, (i * 7) % 18, i % 2, p, dirs)


@njit()
def _njit_heuristic(n, boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs,
                    old_rewards, actions):
    for i in range(n):
        njit_heuristic(boards[i], caps[i][0], caps[i][1], 0, 0, 18, 18, i % 2,
            my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs,
            old_rewards, actions[i][0], actions[i][1], actions[i][2], actions[i][3], caps[i][0], caps[i][1])# Best scenario: No captures
            # old_rewards, actions[i][0], actions[i][1], actions[i][2], actions[i][3], 0, 0)                  # Worst scenario: Always a new capture


@njit()
def _old_njit_heuristic(n, boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs):
    for i in range(n):
        old_njit_heuristic(boards[i], caps[i][0], caps[i][1], 0, 0, 18, 18, i % 2,
            my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs)


def _log(fname, times, ranges):
    print('######################')
    print('heuristic', fname, ': ')
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


def test_find_align_reward(boards, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph):

    times = []
    ranges = test_ranges
    dirs = np.array([
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0]
    ], dtype=np.int32)

    p = np.array([
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    ],
        dtype=np.int32
    )

    _find_align_reward(1, boards, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, dirs, p)
    times.append(time.perf_counter())
    for r in ranges:
        _find_align_reward(r, boards, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, dirs, p)
        times.append(time.perf_counter())

    _log('find align reward', times, ranges)


def test_njit_heuristic(boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs):
    old_rewards = np.zeros((21, 21), dtype=Typing.HeuristicGraphDtype)      # Balek des 10000 c'est du write only. Impact pas le temps
    actions = np.random.randint(0, 19, (10000, 4), dtype=Typing.TupleDtype)

    _njit_heuristic(1, boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs, old_rewards, actions)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _njit_heuristic(r, boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs, old_rewards, actions)
        times.append(time.perf_counter())

    _log('njit heuristic', times, ranges)


def test_old_njit_heuristic(boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs):
    _old_njit_heuristic(1, boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs)

    times = []
    ranges = test_ranges

    times.append(time.perf_counter())
    for r in ranges:
        _old_njit_heuristic(r, boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs)
        times.append(time.perf_counter())

    _log('njit old heuristic', times, ranges)

if __name__ == "__main__":

    pad_boards = np.random.randint(0, 2, (10000, 2, 26, 26), dtype=Typing.BoardDtype)
    boards = np.random.randint(0, 2, (10000, 2, 19, 19), dtype=Typing.BoardDtype)
    caps = np.random.randint(0, 5, (10000, 2), dtype=Typing.TupleDtype)

    my_h_graph = init_my_heuristic_graph()
    opp_h_graph = init_opp_heuristic_graph()
    my_cap_graph = init_my_captures_graph()
    opp_cap_graph = init_opp_captures_graph()
    pows = np.array([
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        ], dtype=Typing.MCTSIntDtype
    )
    dirs = np.array([
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0]
        ], dtype=Typing.MCTSIntDtype
    )

    # test_find_align_reward(pad_boards, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph)
    test_njit_heuristic(boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs)
    test_old_njit_heuristic(boards, caps, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs)

    # valid = heuristics_com)
    # time_benchmark()
    # if valid:
    #     print(f"Heuristics returns same results ! :)")

    # else:
    #     print(f"New heuristics sucks ! :(")