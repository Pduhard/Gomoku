from GomokuLib.Game.Action import GomokuAction
from .AbstractRule import AbstractRule


class GameEndingCapture(AbstractRule):

	def winning(self, action: GomokuAction):
		pass
