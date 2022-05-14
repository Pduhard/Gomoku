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
		self.stats = np.zeros((3, 2), dtype=np.int8)

	def winning(self, action):
		if self.stats[2][self.engine.player_idx ^ 1] == 0:
			return 0

		ar, ac = self.stats[self.engine.player_idx ^ 1]
		gz = self.engine.get_game_zone()

		board = self.engine.state.board
		if not board.flags['C_CONTIGUOUS']:
			board = np.ascontiguousarray(board)
			print("NOT CONTIGUOUS ARRAY !!!!!!")
		c_board = ffi.cast("char *", board.ctypes.data)

		# win = fastcore.basic_rule_winning(c_board, ar, ac)
		win = fastcore.is_winning(c_board, 361, ar, ac, gz[0], gz[1], gz[2], gz[3])

		if win:
			return 3

		self.stats[2][self.engine.player_idx ^ 1] = 0
		return 0

	def nowinning(self, last_action: GomokuAction):
		self.stats[2][self.engine.player_idx] = 1
		self.stats[self.engine.player_idx] = last_action.action
		return True

	def create_snapshot(self):
		return self.stats
		# return {
		# 	'last_capture': [GomokuAction(*c.action) if c is not None else c for c in self.last_capture],
		# 	'check_ending_capture': self.check_ending_capture.copy()
		# }

	def update_from_snapshot(self, stats: np.ndarray):
		self.stats[...] = stats
		# self.last_capture = [GomokuAction(*c.action) if c is not None else c for c in snapshot['last_capture']]
		# self.check_ending_capture = snapshot['check_ending_capture'].copy()


	def update(self, rule: AbstractRule):
		self.stats[...] = rule.stats

	# def update(self, engine: Gomoku, _: AbstractRule):
	# 	rule = GameEndingCapture(engine)
	# 	rule.stats[...] = self.stats
	# 	# rule.last_capture = [
	# 	# 	GomokuAction(*c.action)
	# 	# 	if c is not None
	# 	# 	else c
	# 	# 	for c in self.last_capture
	# 	# ]
	# 	# rule.check_ending_capture = self.check_ending_capture.copy()
	# 	return rule
