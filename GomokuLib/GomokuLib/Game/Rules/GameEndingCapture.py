import fastcore
from fastcore._rules import ffi, lib as fastcore

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
		self.check_ending_capture = [0, 0]

	# def count_align_this_way(self, board, x, y, dx, dy):
	# 	xmax, ymax = self.engine.board_size
	# 	for i in range(4):
	# 		x += dx
	# 		y += dy
	# 		if (x < 0 or x >= xmax or y < 0 or y >= ymax or board[0, x, y] == 0):
	# 			return i
	# 	return 4

	def winning(self, action):
		if self.check_ending_capture[self.engine.player_idx ^ 1] == 0:
			return False

		ar, ac = self.last_capture[self.engine.player_idx ^ 1].action

		board = self.engine.state.board[::-1]
		if not board.flags['C_CONTIGUOUS']:
			board = np.ascontiguousarray(board)
		c_board = ffi.cast("char *", board.ctypes.data)

		# win = fastcore.basic_rule_winning(c_board, ar, ac)
		win = fastcore.is_winning(c_board, ar, ac, *self.engine.game_zone)

		if win:
			raise ForceWinOpponent(reason="GameEndingCapture")

		self.check_ending_capture[self.engine.player_idx ^ 1] = 0
		return False

	def nowinning(self, action: GomokuAction):
		self.last_capture[self.engine.player_idx] = action
		self.check_ending_capture[self.engine.player_idx] = 1
		return True

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
