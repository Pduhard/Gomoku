
from typing import Union

from .Gomoku import Gomoku

class GomokuRunner:

    def __init__(self, board_size: Union[int, tuple[int]] = 19,
                 rules: list[str] = ['Capture', 'Game-Ending-Capture', 'no-double-threes'],
                 **kwargs) -> None:
        self.rules = [r.lower() for r in rules]
        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.engine = Gomoku(board_size,
            is_capture_active=('capture' in self.rules),
            is_game_ending_capture_active=('game-ending-capture' in self.rules),
            is_no_double_threes_active='no-double-threes' in self.rules)

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