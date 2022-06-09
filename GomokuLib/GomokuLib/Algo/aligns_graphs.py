import numpy as np
import GomokuLib.Typing as Typing

import numba as nb

"""
    _:  Empty cell
    #:  Current player stone
    X:  No matters what cell it is

    Current player heuristic
        Indexes:                        01|2345
            5 stones ->                 XX#####

            4 stones + 2 empty cells -> X_####_

            4 stones + 1 empty cells -> X_####X
            4 stones + 1 empty cells -> XX#_###
            4 stones + 1 empty cells -> XX##_##
            4 stones + 1 empty cells -> XX###_#
            4 stones + 1 empty cells -> XX####_

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
    """
        Baisser l'importance des captures ?
    """
    heuristic_coefs_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=Typing.heuristic_graph_nb_dtype
    )
    heuristic_coefs_dict = {
        'capture': 0.25,
        'my_win_possible': 0.5,
        'opp_win_2_turn': -2, # > 2 * my_win_possible
        'my_win_1_turn': 3,     # > opp_win_2_turn
        'opp_win_1_turn': -4,
        'my_win': 6,
        'opp_win': -8,
    }
    return heuristic_coefs_dict

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
    if align[i] == "$":
        return _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + (~player_mark & 0b11)) # 0b01 becomes 0b10 | 0b10 becomes 0b01

    _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b00)    # Can be an empty cells
    _parse_align(graph, player_mark, v, align, i + 1, (p << 2) + 0b11)    # Can be a map edge

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

def init_my_captures_graph():
    """
        When I will be captured
    """

    my_cap_graph = np.zeros(pow(2, 16), Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # Alignments
    _parse_align(my_cap_graph, 0b10, coefs['capture'], "X$##_XX", 0, 0)
    _parse_align(my_cap_graph, 0b10, coefs['capture'], "X_##$XX", 0, 0)

    fill_graph = np.nonzero(my_cap_graph)
    print("Captures heuristic init parse ", len(fill_graph[0]), " alignments")
    return my_cap_graph

def init_opp_captures_graph():
    """
        When opponent will be captured
    """
    opp_cap_graph = np.zeros(pow(2, 16), Typing.HeuristicGraphDtype)
    coefs = _get_heuristic_coefs()

    # Alignments
    _parse_align(opp_cap_graph, 0b01, coefs['capture'], "X$##_XX", 0, 0)
    _parse_align(opp_cap_graph, 0b01, coefs['capture'], "X_##$XX", 0, 0)

    fill_graph = np.nonzero(opp_cap_graph)
    print("Captures heuristic init parse ", len(fill_graph[0]), " alignments")
    return opp_cap_graph


## Init graphs

my_h_graph = init_my_heuristic_graph()
opp_h_graph = init_opp_heuristic_graph()
my_cap_graph = init_my_captures_graph()
opp_cap_graph = init_opp_captures_graph()
