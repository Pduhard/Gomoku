from __future__ import annotations
from copy import deepcopy
from multiprocessing.dummy import Process
from subprocess import call
from time import sleep
from typing import TYPE_CHECKING, Union
import numpy as np
# import pygame
import multiprocessing as mp
import threading
import time

from GomokuLib.Game.Action.GomokuAction import GomokuAction
from GomokuLib.Game.UI.UIManager import UIManager
import GomokuLib
from .Gomoku import Gomoku
if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer

class GomokuGUI(Gomoku):

    def __init__(self,
                 players: Union[list[AbstractPlayer], tuple[AbstractPlayer]] = None,
                 board_size: Union[int, tuple[int]] = 19,
                 win_size: Union[list[int], tuple[int]] = (1500, 1000),
                 *args, **kwargs) -> None:

        self.gui_outqueue = mp.Queue()
        self.gui_inqueue = mp.Queue()

        super().__init__(players, board_size, *args, **kwargs)

        self.gui = UIManager(self, win_size)
        self.gui_proc = Process(target=self.gui, args=(self.gui_outqueue, self.gui_inqueue))
        self.gui_proc.start()

        self.player_action = None

        print("END __init__() Gomokugui\n")

    def update_UI(self, **kwargs):
        """
            All kwargs information will be sent to UIManager with new snapshot
        """
        print(f"New snapshot created")

        self.gui_outqueue.put({
            'code': 'game-snapshot',
            'data': {
                'time': time.time(),
                'snapshot': self.create_snapshot(),
                'ss_data': kwargs
            },
        })

    def next_turn(self, mode: str = "GomokuGUI.run()", before_next_turn_cb=[], **kwargs) -> None:
        """
            All kwargs information will be sent to UIManager
        """
        cb_ret = super().next_turn(before_next_turn_cb=before_next_turn_cb)

        if mode == "GomokuGUI.run()":
            kwargs['p1'] = str(self.players[0])
            kwargs['p2'] = str(self.players[1])

        self.update_UI(
            **cb_ret,
            **kwargs,
            mode=mode,
            captures=self.get_captures()[::-1],
            board=self.state.board,
            turn=self.turn,
            player_idx=self.player_idx,
            winner=self.winner,
        )
        return cb_ret

    def _run(self, players: AbstractPlayer) -> AbstractPlayer:

        while not self.isover():
            self.get_gui_input()

            p = players[self.player_idx]
            player_action = p.play_turn()

            if isinstance(p, GomokuLib.Player.Bot): # Send player data after its turn
                turn_data = p.algo.get_state_data(self)
                self.apply_action(player_action)
                # breakpoint()
                self.next_turn(
                    **turn_data,
                    before_next_turn_cb=[p.algo.get_state_data_after_action]
                )

            else:
                self.apply_action(player_action)
                self.next_turn()


        print(f"Player {self.winner} win.")
        # sleep(5)

    def get_gui_input(self):

        try:
            while True:
                inpt = self.gui_inqueue.get_nowait() # raise Empty Execption

                if inpt['code'] == 'response-player-action':
                    self.player_action = inpt['data']

                elif inpt['code'] == 'shutdown':
                    exit(0)
                
                elif inpt['code'] == 'game-snapshot':
                    breakpoint()
                    self.update_from_snapshot(inpt['data'])
        except:
            pass

    def wait_player_action(self):
        self.gui_outqueue.put({
            'code': 'request-player-action'
        })
        print(f"Wait player action ...")
        while True:
            self.get_gui_input()

            if self.player_action:
                # if self.pause:
                #     self.gui_outqueue.put({
                #         'code': 'request-player-action'
                #     })
                #     self.player_action = None
                #     continue
                action = self.player_action
                self.player_action = None
                return action
