
from time import perf_counter
from typing import Union
import time

from GomokuLib.Game.GameEngine.Snapshot import Snapshot
from .Gomoku import Gomoku


class GomokuRunner:

    def __init__(self, rules: list[str] = ['Capture', 'Game-Ending-Capture', 'no-double-threes'],
                 **kwargs) -> None:
        self.rules = [r.lower() for r in rules]
        is_capture_active = 'capture' in self.rules
        is_game_ending_capture_active = 'game-ending-capture' in self.rules
        is_no_double_threes_active = 'no-double-threes' in self.rules

        self.players = [None, None]
        self.engine = Gomoku(
            is_capture_active,
            is_game_ending_capture_active,
            is_no_double_threes_active,
        )

    def _run(self):

        while not self.engine.isover():

            p = self.players[self.engine.player_idx]
            time_before_turn = perf_counter()

            player_action = p.play_turn(self)

            time_after_turn = perf_counter()
            dtime_turn = int((time_after_turn - time_before_turn) * 1000)
            print(f"Played in {dtime_turn} ms")

            self.engine.apply_action(player_action)
            self.engine.next_turn()
            print(f"Game board ([0] -> p1 / [1] -> p2):\n{self.engine.board}\n")

        print(f"Player {self.engine.winner} win.")

    def run(self, players: list, init_snapshot: int = None, n_games: int = 1):

        self.players = players
        winners = []
        for i in range(n_games):

            self.engine.init_game()
            if init_snapshot:
                Snapshot.update_from_snapshot(self.engine, init_snapshot)

            self._run()
            for p in self.players:
                p.init()

            winner = players[self.engine.winner] if self.engine.winner >= 0 else self.engine.winner
            winners.append(f"P{self.engine.winner}: {str(winner)}")

        return winners
