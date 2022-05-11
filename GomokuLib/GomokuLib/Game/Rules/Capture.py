from typing import Any

import fastcore
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
		self.player_count_capture = [0, 0]

	def endturn(self, action: GomokuAction):

		c_board = ffi.cast("char *", self.engine.state.board.ctypes.data)
		y, x = action.action
		# print(f"NEXT TURN CAPTURE")

		count1 = fastcore.count_captures(c_board, y, x, *self.engine.game_zone)
		self.player_count_capture[self.engine.player_idx] += count1

	def winning(self, action: GomokuAction):
		if self.player_count_capture[self.engine.player_idx] >= 5:
			raise ForceWinPlayer(reason="Five captures.")
		return False

	def get_current_player_captures(self):
		return self.player_count_capture[::-1] if self.engine.player_idx else self.player_count_capture

	def create_snapshot(self):
		return {
			'player_count_capture': self.player_count_capture.copy()
		}
	
	def update_from_snapshot(self, snapshot):
		self.player_count_capture = snapshot['player_count_capture']

	def copy(self, engine: Gomoku, rule: AbstractRule):
		newrule = Capture(engine)
		newrule.player_count_capture = rule.player_count_capture.copy()
		# print("copy: ", newrule.player_count_capture, rule.player_count_capture, rule)
		return newrule
