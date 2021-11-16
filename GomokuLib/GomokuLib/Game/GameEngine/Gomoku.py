
from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..Action.GomokuAction import GomokuAction
    from ..State.GomokuState import GomokuState
    from ...Player.AbstractPlayer import AbstractPlayer

from .AbstractGameEngine import AbstractGameEngine

class Gomoku(AbstractGameEngine):

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]],
                 board_size: Union[int, tuple[int]] = 19, **kwargs) -> None:
        super().__init__(players)
        print('init gomoku: ')
        for p in players:
            print(p.engine)
        self.players = players
        self.board_size = board_size
        self.isover = False
        # init board

    def get_state(self) -> GomokuState:
        pass

    def get_actions(self) -> list[GomokuAction]:
        pass

    def apply_action(self, action: GomokuAction) -> None:
        pass

    def next_turn(self) -> None:
        pass

    def run(self) -> AbstractPlayer:
        # init vars
        self.player_idx = 0
        self.current_player = self.players[0]
        # game loop
        while self.isover is False:
            actions, state = self.get_actions(), self.get_state()
            player_action = self.current_player.play_turn(actions, state)
            self.apply_action(player_action)
            self.next_turn()
        pass

    # def is_endgame(self) -> AbstractPlayer:
    #     pass
