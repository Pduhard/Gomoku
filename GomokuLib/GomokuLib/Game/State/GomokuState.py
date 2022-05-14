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

class GomokuState():

    def __init__(self, board_size : Union[int, tuple]):
        if isinstance(board_size, int):
            board_size = (2, board_size, board_size)
        elif isinstance(board_size, tuple):
            board_size = (2, *board_size)
        else:
            print('error board size')
            exit(0)
        self.board = np.zeros(board_size, dtype=np.int8, order='C')

