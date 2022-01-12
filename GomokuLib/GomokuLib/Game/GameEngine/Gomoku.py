
from __future__ import annotations
from typing import Union, TYPE_CHECKING
import numpy as np

from GomokuLib.Game.Action.AbstractAction import AbstractAction

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
        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.init_game()

    def init_board(self):
        """
            np array but we should go bitboards next
        """
        return np.zeros(self.board_size, dtype=np.int32)
        # return np.random.randint(-1, 2, self.board_size)

    def init_game(self):
        # init vars
        self.isover = False
        self.player_idx = 0
        self.current_player = self.players[0]
        # init board
        self.board = self.init_board()

        pass

    def set_state(self, GomokuState) -> None:
        pass

    def get_state(self) -> GomokuState:
        pass

    def get_actions(self) -> list[GomokuAction]:
        pass

    def is_valid_action(self, action: GomokuAction) -> bool:
        return True

    def apply_action(self, action: GomokuAction) -> None:
        self.board[action.action] = -1 if self.player_idx else 1

    def next_turn(self) -> None:
        self.player_idx ^= 1
        self.current_player = self.players[self.player_idx]
        pass

    def run(self) -> AbstractPlayer:
        # game loop
        while self.isover is False:
            actions, state = self.get_actions(), self.get_state()
            player_action = self.current_player.play_turn(actions, state)
            if player_action not in actions:
                print("player_action not in actions !")
                exit(0)
            self.apply_action(player_action)
            self.next_turn()
        pass

    # def is_endgame(self) -> AbstractPlayer:
    #     pass
