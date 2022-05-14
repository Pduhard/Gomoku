
from __future__ import annotations
from typing import Union, TYPE_CHECKING
import numpy as np

from GomokuLib.Game.Rules import GameEndingCapture, NoDoubleThrees, Capture, BasicRule, RULES
from GomokuLib.Game.Rules import ForceWinOpponent, ForceWinPlayer

from .Gomoku import Gomoku

from ..State.GomokuState import GomokuState

UI = False

class GomokuRunner:

    capture_class: object = Capture

    def __init__(self, board_size: Union[int, tuple[int]] = 19,
                 rules: list[str] = ['Capture', 'Game-Ending Capture', 'no double-threes'],
                 **kwargs) -> None:
        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.engine = Gomoku(board_size, rules)

    def _run(self, players):

        while not self.engine.isover():
            player = players[self.engine.player_idx]
            player_action = player.play_turn(self.engine)

            self.engine.apply_action(player_action)
            self.engine.next_turn()

        print(f"Player {self.engine.winner} win.")

    def run(self, players):

        self.engine.init_game()
        self._run(players)
        return players[self.engine.winner] if self.engine.winner >= 0 else self.engine.winner