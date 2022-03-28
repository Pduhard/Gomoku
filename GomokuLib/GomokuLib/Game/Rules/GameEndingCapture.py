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

	name = 'GameEndingCapture'

	def __init__(self, engine: Gomoku) -> None:
		super().__init__(engine)
		self.last_capture = [None, None]
		self.check_ending_capture = np.array([0, 0])

	def winning(self, action):
		if self.check_ending_capture[self.engine.player_idx ^ 1] == 0:
			return False
		board = self.engine.state.board[::-1]
		print('la ?')
		ar, ac = self.last_capture[self.engine.player_idx ^ 1].action
		print('oui')

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

		self.check_ending_capture[self.engine.player_idx ^ 1] = 0
		return False

	def nowinning(self, action: GomokuAction):
		self.last_capture[self.engine.player_idx] = action
		self.check_ending_capture[self.engine.player_idx] = 1
		return True

	def count_align_this_way(self, board, x, y, dx, dy):
		xmax, ymax = self.engine.board_size
		for i in range(4):
			x += dx
			y += dy
			if (x < 0 or x >= xmax or y < 0 or y >= ymax or board[0, x, y] == 0):
				return i
		return 4
	
	
	def create_snapshot(self):
		return {
			'last_capture': [GomokuAction(*c.action) if c is not None else c for c in self.last_capture],
			'check_ending_capture': self.check_ending_capture.copy()
		}

	def update_from_snapshot(self, snapshot):
		self.last_capture = [GomokuAction(*c.action) if c is not None else c for c in snapshot['last_capture']]
		self.check_ending_capture = snapshot['check_ending_capture'].copy()

	def copy(self, engine: Gomoku, _: AbstractRule):
		rule = GameEndingCapture(engine)
		rule.last_capture = [GomokuAction(*c.action) if c is not None else c for c in self.last_capture]
		rule.check_ending_capture = self.check_ending_capture.copy()
		return rule
