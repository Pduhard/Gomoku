from __future__ import annotations
from typing import Union, TYPE_CHECKING

import pygame
from pygame import event

if TYPE_CHECKING:
    from GomokuLib.Game.Action.AbstractAction import AbstractAction
    from GomokuLib.Game.State.AbstractState import AbstractState

from GomokuLib.Player.AbstractPlayer import AbstractPlayer

class Human(AbstractPlayer):

    def __init__(self, verbose: dict = None) -> None:
        self.verbose = verbose or {}
        # self.panel = HumanCtrlPanel()

    # def handle_click(self):
    #     """

    #     """
    #     pass

    # def handle_event(self):
    #     while True:
    #         for event in pygame.event.get():
    #             if event == CLICK:
    #                 pass
                    


    def play_turn(self, state: AbstractState,
                  actions: list[AbstractAction]) -> AbstractAction:

        return self.engine.wait_player_action()
            
            # if event["type"] == 'hint':
            #     self.game_engine.display_hint()



# class CtrlPanel(metaclass=ABCMeta):
#     """
#         Bot: Logs sur les algorithm du CtrlPanel
#         Human: Hints apportes par les algo du CtrlPanel

#     """
#     def __init__(self, player: AbstractPlayer) -> None:
#         pass
