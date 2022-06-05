
from time import perf_counter
from typing import Union
import time
from .Gomoku import Gomoku

from numba import njit


@njit()
def force_config1(engine):
    # self.engine.init_game()

    engine.apply_action((3, 3))
    engine.next_turn()

    engine.apply_action((4, 3))
    engine.next_turn()

    engine.apply_action((3, 4))
    engine.next_turn()

    engine.apply_action((4, 4))
    engine.next_turn()

    engine.apply_action((3, 5))
    engine.next_turn()


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
            time_before_turn = perf_counter()

            player_action = p.play_turn(self)

            time_after_turn = perf_counter()
            dtime_turn = int((time_after_turn - time_before_turn) * 1000)
            print(f"Played in {dtime_turn} ms")

            self.engine.apply_action(player_action)
            self.engine.next_turn()
            print(f"self.engine.board:\n{self.engine.board}\n")

        print(f"Player {self.engine.winner} win.")

    def run(self, players, init_config: int = None, *args, **kwargs):

        self.engine.init_game()
        if init_config == 1:
            force_config1(self.engine)

        self._run(players, *args, **kwargs)
        return players[self.engine.winner] if self.engine.winner >= 0 else self.engine.winner
