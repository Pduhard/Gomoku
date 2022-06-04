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


@njit()
def get_heuristic_coefs():

    heuristic_coefs_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=Typing.mcts_int_nb_dtype
    )
    heuristic_coefs_dict = {
        'my_win_possible': 2,
        'opp_win_2_turn': -4,
        'my_win_1_turn': 5,
        'opp_win_1_turn': -5,
        'my_win': 6,
        'opp_win': -6,
    }
    return heuristic_coefs_dict


@njit()
def create_idx(graph, player_mark, v, align, i, p):
    """
        create_idx() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments

        Compute index p of array graph using 14 bits
            2 bits per cell, representing empty cell    '_' = 0b00
                                    cell with my stone  '#' = 0b10
                    and cell state that doesn't matter  'X' = 0b00/0b01/0b10
    """
    # print(f"v, align, i, p = ", v, align, i, p)
    if i == 7:
        if graph[p]:
            print(f"Already a reward here !!", align, p, v, " overwrite ", graph[p])
        # print(f"graph[p] = v / graph[{p}] = {v}")
        graph[p] = v
        return 

    if align[i] == "_":
        return create_idx(graph, player_mark, v, align, i + 1, (p << 2) + 0b00)
    if align[i] == "#":
        return create_idx(graph, player_mark, v, align, i + 1, (p << 2) + player_mark)

    create_idx(graph, player_mark, v, align, i + 1, (p << 2) + 0b00)    # Can be an empty cells
    create_idx(graph, player_mark, v, align, i + 1, (p << 2) + 0b11)    # Can be a map edge

    # create_idx(graph, player_mark, v, align, i + 1, (p << 2) + 0b01)  # Can be an opponent's stone
    # create_idx(graph, player_mark, v, align, i + 1, (p << 2) + 0b10)  # Can be an opponent's stone

    if player_mark == 0b10: # Prevent double rewards from one alignment
        create_idx(graph, player_mark, v, align, i + 1, (p << 2) + 0b01)  # Can be an opponent's stone
    else:
        create_idx(graph, player_mark, v, align, i + 1, (p << 2) + 0b10)  # Can be an opponent's stone


@njit()
def init_my_heuristic_graph():
    """
        create_idx() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments
    """

    my_graph = np.zeros(pow(2, 16), Typing.HeuristicGraphDtype)
    coefs = get_heuristic_coefs()

    print("init_my_heuristic_graph", len(my_graph), my_graph)
    # Current player alignments
    create_idx(my_graph, 0b10, coefs['my_win_possible'], "__###_X", 0, 0)
    create_idx(my_graph, 0b10, coefs['my_win_possible'], "X_#_##_", 0, 0)
    create_idx(my_graph, 0b10, coefs['my_win_possible'], "X_##_#_", 0, 0)
    create_idx(my_graph, 0b10, coefs['my_win_possible'], "X_###__", 0, 0)

    create_idx(my_graph, 0b10, coefs['my_win_1_turn'], "X_####_", 0, 0)

    create_idx(my_graph, 0b10, coefs['my_win'], "XX#####", 0, 0)

    fill_graph = np.nonzero(my_graph)
    print(fill_graph)
    print("Length: ", len(fill_graph[0]))
    return my_graph


@njit()
def init_opp_heuristic_graph():
    """
        create_idx() calls need to be in ascending order according to the rewards.
            Because some of them overwrite old registered alignments
    """

    opp_graph = np.zeros(pow(2, 16), Typing.HeuristicGraphDtype)
    coefs = get_heuristic_coefs()

    print("init_opp_heuristic_graph", len(opp_graph), opp_graph)
    # Opponent alignments
    create_idx(opp_graph, 0b01, coefs['opp_win_2_turn'], "__###_X", 0, 0)
    create_idx(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_#_##_", 0, 0)
    create_idx(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_##_#_", 0, 0)
    create_idx(opp_graph, 0b01, coefs['opp_win_2_turn'], "X_###__", 0, 0)

    create_idx(opp_graph, 0b01, coefs['opp_win_1_turn'], "X_####X", 0, 0)
    create_idx(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX#_###", 0, 0)
    create_idx(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX##_##", 0, 0)
    create_idx(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX###_#", 0, 0)
    create_idx(opp_graph, 0b01, coefs['opp_win_1_turn'], "XX####_", 0, 0)

    create_idx(opp_graph, 0b01, coefs['opp_win'], "XX#####", 0, 0)

    fill_graph = np.nonzero(opp_graph)
    print(fill_graph)
    print("Length: ", len(fill_graph[0]))
    return opp_graph


if __name__ == "__main__":
    print(init_my_heuristic_graph())
    print(init_opp_heuristic_graph())
