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

            4 stones + 1 empty cells -> X_####X     my_win
            4 stones + 1 empty cells -> XX#_###
            4 stones + 1 empty cells -> XX##_##
            4 stones + 1 empty cells -> XX###_#
            4 stones + 1 empty cells -> XX####_

            3 stones + 3 empty cells -> __###_X     my_win_1_turn
            3 stones + 3 empty cells -> X_#_##_
            3 stones + 3 empty cells -> X_##_#_
            3 stones + 3 empty cells -> X_###__

    Opponent player heuristic:
        Indexes:                        01|2345
            5 stones ->                 XX#####     opp_win
            4 stones + 2 empty cells -> X_####_     opp_win

            4 stones + 1 empty cells -> X_####X     opp_win_1_turn
            4 stones + 1 empty cells -> XX#_###
            4 stones + 1 empty cells -> XX##_##
            4 stones + 1 empty cells -> XX###_#
            4 stones + 1 empty cells -> XX####_

            3 stones + 3 empty cells -> __###_X     opp_win_2_turn
            3 stones + 3 empty cells -> X_#_##_
            3 stones + 3 empty cells -> X_##_#_
            3 stones + 3 empty cells -> X_###__
"""

"""
    Initialization of heuristic's data
"""

@njit()
def _get_heuristic_coefs():
    heuristic_coefs_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=Typing.heuristic_graph_nb_dtype
    )
    heuristic_coefs_dict = {    # Same coef in hpruning ! Be careful
        'capture': 0.5,

        'opp_win_2_turn': -1,
        'my_win_1_turn': 2,
        'opp_force_countering': -4,

        'opp_win_1_turn': -6,
        'my_win': 7,
        'opp_win': -8,
    }
    return heuristic_coefs_dict

@njit()
def _parse_align(graph, player_mark, v, align, i, p):
    """
        '#' needs to be at index 2. Because that's how the heuristic test

        _parse_align() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments

        Compute index p of array graph using 14 bits
            2 bits per cell, representing empty cell    '_' = 0b00
                                    cell with my stone  '#' = 0b10
                    and cell state that doesn't matter  'X' = 0b00/0b01/0b10
    """
    if i == 7:
        graph[p] = v
        return

    if align[i] == "_":
        return _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b00)
    if align[i] == "#":
        return _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + player_mark)
    if align[i] == "$":
        return _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + (~player_mark & 0b11)) # 0b01 becomes 0b10 | 0b10 becomes 0b01

    _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b00)    # Can be an empty cells
    _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b11)    # Can be a map edge

    if player_mark == 0b10: # Prevent double rewards from one alignment
        _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b01)  # Can be an opponent's stone
    else:
        _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b10)  # Can be an opponent's stone


@njit()
def init_my_heuristic_graph():
    """
        _parse_align() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments
    """

    my_graph = np.zeros(2 << 16, Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # Current player alignments
    _parse_align(my_graph, 0b10, coefs['my_win_1_turn'], "__###_X", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_1_turn'], "X_#_##_", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_1_turn'], "X_##_#_", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win_1_turn'], "X_###__", 0, 0)

    _parse_align(my_graph, 0b10, coefs['my_win'], "X_####X", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win'], "XX#_###", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win'], "XX##_##", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win'], "XX###_#", 0, 0)
    _parse_align(my_graph, 0b10, coefs['my_win'], "XX####_", 0, 0)

    return my_graph

@njit()
def init_opp_heuristic_graph():
    """
        _parse_align() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments
    """

    opp_graph = np.zeros(2 << 16, Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # Opponent alignments
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "__###_X", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_#_##_", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_##_#_", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_###__", 0, 0)

    _parse_align(opp_graph, 0b01, coefs['opp_force_countering'], "X_####X", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_force_countering'], "XX#_###", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_force_countering'], "XX##_##", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_force_countering'], "XX###_#", 0, 0)
    _parse_align(opp_graph, 0b01, coefs['opp_force_countering'], "XX####_", 0, 0)

    _parse_align(opp_graph, 0b10, coefs['opp_win_1_turn'], "X_####_", 0, 0)

    _parse_align(opp_graph, 0b01, coefs['opp_win'], "XX#####", 0, 0)

    return opp_graph

@njit()
def init_my_captures_graph():
    """
        When I will be captured
    """

    my_cap_graph = np.zeros(2 << 16, Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # Current player captures alignments
    _parse_align(my_cap_graph, 0b10, coefs['capture'], "X$##_XX", 0, 0)
    _parse_align(my_cap_graph, 0b10, coefs['capture'], "X_##$XX", 0, 0)

    return my_cap_graph

@njit()
def init_opp_captures_graph():
    """
        When opponent will be captured
    """
    opp_cap_graph = np.zeros(2 << 16, Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # Opponent player captures alignments
    _parse_align(opp_cap_graph, 0b01, coefs['capture'], "X$##_XX", 0, 0)
    _parse_align(opp_cap_graph, 0b01, coefs['capture'], "X_##$XX", 0, 0)

    return opp_cap_graph
