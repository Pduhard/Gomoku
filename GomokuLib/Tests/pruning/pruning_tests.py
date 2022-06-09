import GomokuLib
from GomokuLib.Algo.hpruning import njit_hpruning, _create_align_masks
import GomokuLib.Typing as Typing 
from GomokuLib.Algo.aligns_graphs import my_h_graph, opp_h_graph

from time import perf_counter
import numpy as np
import numba as nb

from numba import njit


def generate_rd_boards(n, mode):

    if mode == 0:
        board = np.zeros((n, 2, 19, 19), dtype=Typing.BoardDtype)
    
    elif mode == 1:
        board = np.random.randint(0, 2, (n, 2, 19, 19), dtype=Typing.BoardDtype)
    
    elif mode == 2:
        board = np.ones((n, 2, 19, 19), dtype=Typing.BoardDtype)

    for loop in range(n):
        for y in range(19):
            for x in range(19):
                if board[loop, 0, y, x] and board[loop, 1, y, x]:
                    board[loop, 1, y, x] = Typing.BoardDtype(0)
    return board.astype(Typing.BoardDtype)


def debug_njit_hpruning():

    boards = generate_rd_boards(10, 1)
    boards[0] = np.zeros((2, 19, 19), dtype=Typing.PruningDtype)
    for loop in range(0, 10):

        # board = np.zeros((2, 26, 26), dtype=Typing.BoardDtype)
        # board = np.zeros((2, 19, 19), dtype=Typing.BoardDtype)

        # board[0, 4, 13] = 1
        # board[0, 4, 15] = 1
        # board[0, 4, 16] = 1

        # board[1, 3, 14] = 1
        # board[1, 3, 15] = 1
        # board[1, 3, 16] = 1
        # print(board.shape)
        # print("Board:\n", board)

        board = boards[loop]

        mask = njit_hpruning(board, 0, 0, 18, 18)
        # mask = _create_align_masks(board, my_h_graph, 4, 13)
        # mask = _create_align_masks(board, opp_h_graph, 3, 14)

        print("\nRESULTS:", board, "\n", mask.shape, mask)


if __name__ == "__main__":
    debug_njit_hpruning()
