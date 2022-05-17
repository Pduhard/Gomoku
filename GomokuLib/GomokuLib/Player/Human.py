from __future__ import annotations
from GomokuLib.Game.GameEngine.GomokuRunner import GomokuRunner


class Human:

    def __str__(self):
        return f"Human"

    def play_turn(self, runner: GomokuRunner, **kwargs) -> tuple[int]:
        return runner.wait_player_action()
