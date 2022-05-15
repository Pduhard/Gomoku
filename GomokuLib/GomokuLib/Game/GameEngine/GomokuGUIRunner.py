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

from GomokuLib.Game.UI.UIManager import UIManager
import GomokuLib
from .GomokuRunner import GomokuRunner

class GomokuGUIRunner(GomokuRunner):

    def __init__(self, win_size: Union[list[int], tuple[int]] = (1500, 1000),
                 *args, **kwargs) -> None:

        super().__init__()
        self.gui_outqueue = mp.Queue()
        self.gui_inqueue = mp.Queue()

        # self.engine is defined in super
        self.gui = UIManager(self.engine, win_size)
        self.gui_proc = Process(target=self.gui, args=(self.gui_outqueue, self.gui_inqueue))
        self.gui_proc.start()

        self.player_action = None

        print("END __init__() GomokuguiRunner\n")

    def update_UI(self, **kwargs):
        """
            All kwargs information will be sent to UIManager with new snapshot
        """
        print(f"New snapshot created")

        self.gui_outqueue.put({
            'code': 'game-snapshot',
            'data': {
                'time': time.time(),
                'snapshot': self.engine.create_snapshot(),
                'ss_data': kwargs
            },
        })

    def _run(self, players, mode: str = "GomokuGUIRunner.run()"):

        while not self.engine.isover():
            self.get_gui_input()

            p = players[self.engine.player_idx]
            player_action = p.play_turn(self.engine)

            if isinstance(p, GomokuLib.Player.Bot): # Send player data after its turn
                turn_data = p.algo.get_state_data(self.engine)
                self.engine.apply_action(player_action)
                self.engine._next_turn_rules()
                turn_data.update(p.algo.get_state_data_after_action(self.engine))
                self.engine._shift_board()
                if mode == "GomokuGUIRunner.run()":
                    turn_data['p1'] = str(players[0])
                    turn_data['p2'] = str(players[1])
                # breakpoint()
                self.update_UI(
                    **turn_data,
                    mode=mode,
                    captures=self.engine.get_captures()[::-1],
                    board=self.engine.board,
                    turn=self.engine.turn,
                    player_idx=self.engine.player_idx,
                    winner=self.engine.winner,
                )

            else:
                self.engine.apply_action(player_action)
                self.engine.next_turn()
            # print(f"Game zone: {self.game_zone[0]} {self.game_zone[1]} into {self.game_zone[2]} {self.game_zone[3]}")


        print(f"Player {self.engine.winner} win.")
        # sleep(5)

    def get_gui_input(self):

        try:
            while True:
                inpt = self.gui_inqueue.get_nowait() # raise Empty Execption

                if inpt['code'] == 'response-player-action':
                    ar, ac = inpt['data']
                    self.player_action = (ar, ac)

                elif inpt['code'] == 'shutdown':
                    exit(0)
                
                elif inpt['code'] == 'game-snapshot':
                    breakpoint()
                    self.engine.update_from_snapshot(inpt['data'])
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
