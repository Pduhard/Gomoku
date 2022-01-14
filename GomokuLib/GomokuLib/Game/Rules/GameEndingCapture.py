from time import perf_counter
from GomokuLib.Game.Action import GomokuAction
import numpy as np

from GomokuLib.Game.GameEngine import Gomoku

from .AbstractRule import AbstractRule



class ForceWinPlayer(Exception):
    def __init__(self, reason="No reason", *args: object) -> None:
        super().__init__(args)
        self.reason = reason

class ForceWinOpponent(Exception):
    def __init__(self, reason="No reason", *args: object) -> None:
        super().__init__(args)
        self.reason = reason

class GameEndingCapture(AbstractRule):

	def __init__(self, engine: Gomoku) -> None:
		super().__init__(engine)
		self.last_capture = [None for _ in self.engine.players]
		self.winning = None

	def win_again(self, action):
		
		print("Win again ???")
		board = self.engine.state.board[::-1]

		ar, ac = self.last_capture[self.engine.player_idx ^ 1].action

		win = False
		if (self.count_align_this_way(board, ar, ac, -1, -1) +\
			self.count_align_this_way(board, ar, ac, 1, 1) + 1 >= 5):
			win = True

		elif (self.count_align_this_way(board, ar, ac, -1, 0) +\
			self.count_align_this_way(board, ar, ac, 1, 0) + 1 >= 5):
			win = True

		elif (self.count_align_this_way(board, ar, ac, -1, 1) +\
			self.count_align_this_way(board, ar, ac, 1, -1) + 1 >= 5):
			win = True

		elif (self.count_align_this_way(board, ar, ac, 0, -1) +\
			self.count_align_this_way(board, ar, ac, 0, 1) + 1 >= 5):
			win = True

		if win:
			raise ForceWinOpponent(reason="GameEndingCapture")

		self.engine.remove_rule(self, "winning")
		self.winning = None
		print(f"remove rule player {self.engine.player_idx ^ 1}")
		return False

	def nowinning(self, action: GomokuAction):

		self.last_capture[self.engine.player_idx] = action
		self.winning = self.win_again
		self.engine.new_rule(self, "winning")
		print(f"add rule player {self.engine.player_idx}")
		return True

	def count_align_this_way(self, board, x, y, dx, dy):
		xmax, ymax = self.engine.board_size
		for i in range(4):
			x += dx
			y += dy
			if (x < 0 or x >= xmax or y < 0 or y >= ymax or board[0, x, y] == 0):
				return i
		return 4
