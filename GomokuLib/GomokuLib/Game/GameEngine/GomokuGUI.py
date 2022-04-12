from __future__ import annotations
from copy import deepcopy
from multiprocessing.dummy import Process
from time import sleep
from typing import TYPE_CHECKING, Union
import numpy as np
# import pygame
import multiprocessing as mp
import threading

from GomokuLib.Game.Action.GomokuAction import GomokuAction
from GomokuLib.Game.UI.UIManager import UIManager

from .Gomoku import Gomoku
if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer

class GomokuGUI(Gomoku):

    def __init__(self,
                 players: Union[list[AbstractPlayer], tuple[AbstractPlayer]],
                 board_size: Union[int, tuple[int]] = 19,
                 win_size: Union[list[int], tuple[int]] = (1500, 1000),
                 history_size: int = 0,
                 **kwargs) -> None:
        super().__init__(players, board_size=board_size, history_size=history_size, **kwargs)

        self.gui_outqueue = mp.Queue()
        self.gui_inqueue = mp.Queue()
        self.gui = UIManager(win_size, self.board_size)
        self.gui_proc = Process(target=self.gui, args=(self.gui_outqueue, self.gui_inqueue))
        self.gui_proc.start()

        self.pause = False
        self.shutdown = False
        self.player_action = None

        # self.processes = [
        #     self.gui_proc,
        # ]
        # threading.Thread(target=self.gui, args=(self.get_turn_data,))

        # self.UI.drawUI(board=self.state.board, player_idx=self.player_idx)

        # self.initUI(win_size)
        # self.drawUI()
        print("END __init__() Gomokugui\n")

    def update_UI(self, **kwargs):
        kwargs.update({'snapshot': self.create_snapshot()})
        self.gui_outqueue.put({
            'code': 'game-snapshot',
            'data': kwargs,
        })


    def _run(self, players: AbstractPlayer) -> AbstractPlayer:

        while not self.isover():
            self.get_gui_input()
            # events = self.gui.get_events()
            # self.apply_events(events)
            if not self.pause:
                self._run_turn(players)
            # self.UI.drawUI(board=self.state.board, player_idx=self.player_idx)
            # self.drawUI()

        print(f"Player {self.winner} win.")
        sleep(5)

    def apply_events(self, events: list):
        pass

    def get_turn_data(self) -> dict:
        return {
            'board': self.state.board,
            'player_idx': self.player_idx
        }

    def get_gui_input(self):

        try:
            while True:
                inpt = self.gui_inqueue.get_nowait()
                if inpt['code'] == 'request-pause':
                    self.pause = inpt['data']
                
                # skip player action

                elif inpt['code'] == 'response-player-action':
                    self.player_action = inpt['data']

                elif inpt['code'] == 'shutdown':
                    self.shutdown = True
                
                elif inpt['code'] == 'game-snapshot':
                    breakpoint()
                    self.update_from_snapshot(inpt['data'])
        except:
            pass

    def wait_player_action(self):
        self.gui_outqueue.put({
            'code': 'request-player-action'
        })
        while True:
            self.get_gui_input()
            if self.shutdown:
                exit(0)
            if self.player_action:
                if self.pause:
                    self.gui_outqueue.put({
                        'code': 'request-player-action'
                    })
                    self.player_action = None
                    continue
                action = self.player_action
                self.player_action = None
                return action
