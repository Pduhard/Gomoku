import numpy as np
import GomokuLib.Typing as Typing

import numba as nb
from numba import njit

"""
    _:  Empty cell
    #:  Current player stone
    X:  No matters what cell it is

    Current player heuristic
        Indexes:                        01|2345
            5 stones ->                 XX#####

            4 stones + 1 empty cells -> X_####_

            3 stones + 3 empty cells -> __###_X
            3 stones + 3 empty cells -> X_#_##_
            3 stones + 3 empty cells -> X_##_#_
            3 stones + 3 empty cells -> X_###__

    Opponent player heuristic:
        Indexes:                        01|2345
            5 stones ->                 XX#####

            4 stones + 1 empty cells -> X_####X
            4 stones + 1 empty cells -> XX#_###
            4 stones + 1 empty cells -> XX##_##
            4 stones + 1 empty cells -> XX###_#
            4 stones + 1 empty cells -> XX####_

            3 stones + 3 empty cells -> __###_X
            3 stones + 3 empty cells -> X_#_##_
            3 stones + 3 empty cells -> X_##_#_
            3 stones + 3 empty cells -> X_###__
"""

"""
    Initialization of heuristic's data
"""

def _get_heuristic_coefs():

    heuristic_coefs_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=Typing.mcts_int_nb_dtype
    )
    heuristic_coefs_dict = {
        'my_win_possible': 0.5,
        'opp_win_2_turn': -1.5, # > 2 * my_win_possible
        'my_win_1_turn': 2,     # > opp_win_2_turn
        'opp_win_1_turn': -4,
        'my_win': 5,
        'opp_win': -6,
    }
    return heuristic_coefs_dict

def _parse_align(graph, player_mark, v, align, i, p):
    """
        _parse_align() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments

        Compute index p of array graph using 14 bits
            2 bits per cell, representing empty cell    '_' = 0b00
                                    cell with my stone  '#' = 0b10
                    and cell state that doesn't matter  'X' = 0b00/0b01/0b10
    """
    # print(f"v, align, i, p = ", v, align, i, p)
    if i == 7:
        # if graph[p]:
        #     print(f"Already a reward here !!", align, p, v, " overwrite ", graph[p])
        # print(f"graph[p] = v / graph[{p}] = {v}")
        graph[p] = v
        return 

    if align[i] == "_":
        return _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b00)
    if align[i] == "#":
        return _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + player_mark)

    _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b00)    # Can be an empty cells
    _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b11)    # Can be a map edge

    # _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b01)  # Can be an opponent's stone
    # _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b10)  # Can be an opponent's stone

    if player_mark == 0b10: # Prevent double rewards from one alignment
        _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b01)  # Can be an opponent's stone
    else:
        _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b10)  # Can be an opponent's stone

def init_my_heuristic_graph():
    """
        _parse_align() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments
    """

    my_graph = np.zeros(pow(2, 16), Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # print("init_my_heuristic_graph", len(my_graph), my_graph)
    # Current player alignments
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "__###_X", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "X_#_##_", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "X_##_#_", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "X_###__", 0, 0)
    
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "X_####X", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "XX#_###", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "XX##_##", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "XX###_#", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_possible'], "XX####_", 0, 0)

    _parse_align(my_graph, 0b10, coefs['my_win_1_turn'], "X_####_", 0, 0)

    _parse_align(my_graph, 0b10, coefs['my_win'], "XX#####", 0, 0)

    fill_graph = np.nonzero(my_graph)
    # print(fill_graph)
    print("My heuristic init parse ", len(fill_graph[0]), " alignments")
    return my_graph

def init_opp_heuristic_graph():
    """
        _parse_align() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments
    """

    opp_graph = np.zeros(pow(2, 16), Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # print("init_opp_heuristic_graph", len(opp_graph), opp_graph)
    # Opponent alignments
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "__###_X", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_#_##_", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_##_#_", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_###__", 0, 0)

    _parse_align(opp_graph, 0b01, coefs['opp_win_1_turn'], "X_####X", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX#_###", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX##_##", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX###_#", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX####_", 0, 0)

    _parse_align(opp_graph, 0b01, coefs['opp_win'], "XX#####", 0, 0)

    fill_graph = np.nonzero(opp_graph)
    # print(fill_graph)
    print("Opponent heuristic init parse ", len(fill_graph[0]), " alignments")
    return opp_graph


"""
    Heuristic computation
"""

@njit()
def _old_find_align_reward(board, graph, sr, sc):
    dirs = [
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0]
    ]

    p = np.array(
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        dtype=np.float32    # np.dot handle by Numba require float
    )
    buf = np.zeros(14, dtype=np.float32)
    rewards = np.zeros(4, dtype=np.int32)

    for di in range(4):

        dr, dc = dirs[di]
        r, c = sr - 2 * dr, sc - 2 * dc
        for i in range(0, 14, 2):
            buf[i:i + 2] = board[:, r, c]
            r += dr
            c += dc

        graph_id = np.dot(buf, p)
        rewards[di] = graph[np.int32(graph_id)]

    return np.sum(rewards)

@njit()
def _find_align_reward(board, graph, sr, sc):
    dirs = [
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0]
    ]

    p = np.array([
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    ],
        dtype=np.int32
    )
    align_ids = np.zeros(4, dtype=np.int32)
    buf = np.zeros((4, 14), dtype=np.int32)
    rewards = np.zeros(4, dtype=np.int32)

    for di in range(4):

        dr, dc = dirs[di]
        r, c = sr - 2 * dr, sc - 2 * dc
        for i in range(0, 14, 2):
            buf[di, i:i + 2] = board[:, r, c]
            r += dr
            c += dc

    # Compute each digit with its factor
    mul = buf * p

    # Sum all digits to create indexes
    align_ids[:] = np.sum(mul, axis=-1)

    for di in range(4):
        rewards[di] = graph[align_ids[di]]

    return np.sum(rewards)

@njit()
def _compute_capture_coef(my_cap, opp_cap):
    return 2 * (my_cap - opp_cap) / (5.5 - max(my_cap, opp_cap))

@njit()
def njit_heuristic(board, my_graph, opp_graph, c0, c1, gz_start_r, gz_start_c, gz_end_r, gz_end_c):

    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    rewards = np.zeros((21, 21), dtype=np.int32)
    for y in range(2 + gz_start_r, 2 + gz_end_r):
        for x in range(2 + gz_start_c, 2 + gz_end_c):

            if board_pad[0, y, x]:
                rewards[y, x] = _find_align_reward(board_pad, my_graph, y, x)

            elif board_pad[1, y, x]:
                rewards[y, x] = _find_align_reward(board_pad, opp_graph, y, x)

    # print("All rewards ->" ,rewards)
    x = np.sum(rewards) + _compute_capture_coef(c0, c1)
    return 1 / (1 + np.exp(-0.4 * x))

@njit()
def old_njit_heuristic(board, my_graph, opp_graph, c0, c1, gz_start_r, gz_start_c, gz_end_r, gz_end_c):

    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    rewards = np.zeros((21, 21), dtype=np.int32)
    for y in range(2 + gz_start_r, 2 + gz_end_r):
        for x in range(2 + gz_start_c, 2 + gz_end_c):

            if board_pad[0, y, x]:
                rewards[y, x] = _old_find_align_reward(board_pad, my_graph, y, x)

            elif board_pad[1, y, x]:
                rewards[y, x] = _old_find_align_reward(board_pad, opp_graph, y, x)

    # print("All rewards ->" ,rewards)
    x = np.sum(rewards) + _compute_capture_coef(c0, c1)
    return 1 / (1 + np.exp(-0.4 * x))



if __name__ == "__main__":
    print(init_my_heuristic_graph())
    print(init_opp_heuristic_graph())
