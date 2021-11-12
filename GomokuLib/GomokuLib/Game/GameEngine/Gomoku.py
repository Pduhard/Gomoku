
from typing import Union
from GomokuLib.Game.GameEngine.AbstractGameEngine import AbstractGameEngine
from GomokuLib.Game.Action.GomokuAction import GomokuAction
from GomokuLib.Game.State.GomokuState import GomokuState
from GomokuLib.Player.AbstractPlayer import AbstractPlayer


class Gomoku(AbstractGameEngine):

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]],
                 board_size: Union[int, tuple[int]] = 19, **kwargs) -> None:
        super().__init__(players)
        print('init gomoku: ')
        for p in players:
            print(p.engine)
        # init board

    def get_state(self) -> GomokuState:
        pass

    def get_actions(self) -> list[GomokuAction]:
        pass

    def apply_action(self, action: GomokuAction) -> None:
        pass

    def run(self) -> AbstractPlayer:
        pass

    # def is_endgame(self) -> AbstractPlayer:
    #     pass
