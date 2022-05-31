
from time import perf_counter
from typing import Union
import time
from .Gomoku import Gomoku

class GomokuRunner:

    def __init__(self, rules: list[str] = ['Capture', 'Game-Ending-Capture', 'no-double-threes'],
                 **kwargs) -> None:
        self.rules = [r.lower() for r in rules]
        is_capture_active = 'capture' in self.rules
        is_game_ending_capture_active = 'game-ending-capture' in self.rules
        is_no_double_threes_active = 'no-double-threes' in self.rules

        self.engine = Gomoku(
            is_capture_active,
            is_game_ending_capture_active,
            is_no_double_threes_active,
        )

    def _run(self, players):

        while not self.engine.isover():
            p = players[self.engine.player_idx]
            ts = perf_counter()
            player_action = p.play_turn(self)
            ta = perf_counter()
            print(f"Played in {(ta - ts) * 1000}")

            self.engine.apply_action(player_action)
            self.engine.next_turn()
            print(f"self.engine.board:\n{self.engine.board}\n")

        print(f"Player {self.engine.winner} win.")

    def run(self, players, *args, **kwargs):

        self.engine.init_game()
        self._run(players, *args, **kwargs)
        return players[self.engine.winner] if self.engine.winner >= 0 else self.engine.winner