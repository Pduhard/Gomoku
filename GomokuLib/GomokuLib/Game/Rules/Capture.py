from typing import Any

import fastcore
import numpy as np
from fastcore._rules import ffi, lib as fastcore

from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib.Game.Action import GomokuAction
from GomokuLib.Game.Rules.AbstractRule import AbstractRule
from GomokuLib.Game.Rules.GameEndingCapture import ForceWinPlayer


class Capture(AbstractRule):

	name = 'Capture'

	def __init__(self, engine: Any) -> None:
		super().__init__(engine)
		# self.CAPTURE_MASK=init_capture_mask()
		self.player_count_capture = np.array((0, 0), np.int8)

	def endturn(self, action: GomokuAction):

		c_board = ffi.cast("char *", self.engine.state.board.ctypes.data)
		ar, ac = action.action
		gz = self.engine.get_game_zone()
		# print(f"NEXT TURN CAPTURE")

		count1 = fastcore.count_captures(c_board, ar, ac, gz[0], gz[1], gz[2], gz[3])
		self.player_count_capture[self.engine.player_idx] += count1

	def winning(self, action: GomokuAction):
		if self.player_count_capture[self.engine.player_idx] >= 5:
			return 2
		return 0

	def get_current_player_captures(self, *args):
		return self.player_count_capture[::-1] if self.engine.player_idx else self.player_count_capture

	def create_snapshot(self):
		return self.player_count_capture.copy()
		# return {
		# 	'player_count_capture': self.player_count_capture.copy()
		# }
	
	def update_from_snapshot(self, player_count_capture: np.ndarray):
		self.player_count_capture[:] = player_count_capture
		# self.player_count_capture[:] = snapshot['player_count_capture']

	def update(self, rule: AbstractRule):
		self.player_count_capture[:] = rule.player_count_capture

	# def update(self, engine: Gomoku, rule: AbstractRule):
	# 	newrule = Capture(engine)
	# 	newrule.player_count_capture[:] = rule.player_count_capture
	# 	return newrule
