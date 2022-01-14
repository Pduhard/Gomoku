from typing import Union
from GomokuLib.Game.State.AbstractState import AbstractState
from bitarray import bitarray
import numpy as np

"""

def haswon(board):
	print(board)
	print(board >> 7)
	y = board & (board >> 7)
	print(y)
	print((y >> 2 * 7))
	if (y & (y >> 2 * 7)):
		return 1
	y = board & (board >> 8)
	if (y & (y >> 2 * 8)):
		return 1
	y = board & (board >> 9)
	if (y & (y >> 2 * 9)):
		return 1
	y = board & (board >> 1)
	if (y & (y >> 2)):
		return 1
	return 0
"""

class GomokuState(AbstractState):

    def __init__(self, board_size : Union[int, tuple]):
        if isinstance(board_size, int):
            board_size = (2, board_size, board_size)
        elif isinstance(board_size, tuple):
            board_size = (2, *board_size)
        else:
            print('error board size')
            exit(0)
        self.board = np.zeros(board_size, dtype=np.int32)
        self._full_board_uptodate = False

    @property
    def board(self):
        print("getter")
        return self._board

    @board.setter
    def board(self, value):
        print("setter")
        self._board = value
        self._full_board_uptodate = False

    @property
    def full_board(self):
        if not self._full_board_uptodate:
            self._full_board = self._board[0] | self._board[1]
            self._full_board_uptodate = True
        return self._full_board
