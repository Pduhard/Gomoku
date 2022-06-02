from ast import While
from subprocess import call
from typing import Union
from time import sleep
import numpy as np
import pygame
import time

from ..GameEngine.Gomoku import Gomoku
from ..GameEngine.Snapshot import Snapshot

from .Board import Board
from .Button import Button
from .Display import Display

from GomokuLib.Sockets.UISocketClient import UISocketClient


class UIManagerSocket:

    def __init__(self, win_size: tuple, host: str = None, port: int = None):

        self.win_size = win_size
        self.host = host
        self.port = port

        self.engine = Gomoku()
        self.board_size = self.engine.board_size

    def __call__(self): # Thread function

        print("UIManager __call__()\n")
        self.init()

        self.cross_shutdown = False
        while not self.cross_shutdown:

            self.fetch_input()

            self.process_events()
            self.process_inputs()
            self.update()

            self.uisock.send_all()

    def init(self):

        self.callbacks = {}
        self.current_snapshot_idx = -1
        self.snapshot_idx_modified = False
        self.pause = False
        self.game_data = {}
        self.init_time = time.time()
        self.uisock = UISocketClient(host=self.host, port=self.port, name="UIManager")
        self.request_player_action = False
        self.pause = False
        self.board_clicked_action = None
        self.inputs = []
        self.game_snapshots = []
        self.snapshot_idx_modified = False

        self.initUI(self.win_size)
        
    def initUI(self, win_size: Union[list[int], tuple[int]]):

        print(f"board_size: {self.board_size}")
        assert len(win_size) == 2

        print("init GUI")
        pygame.init()

        self.win = pygame.display.set_mode(win_size)
        # self.win.convert_alpha()

        self.main_board = Board(self.win, origin=(0, 0), size=(950, 950), board_size=self.board_size)
        self.components = [
            self.main_board,
            Display(self.win, origin=(1000, 350), size=(450, 600)),
            Button(self.win, origin=(1050, 100), size=(100, 100), event_code='step-back', color=(0, 255, 255)),
            Button(self.win, origin=(1200, 100), size=(100, 100), event_code='pause-play', color=(0, 255, 0)),
            Button(self.win, origin=(1350, 100), size=(100, 100), event_code='step-front', color=(0, 255, 255)),
            Button(self.win, origin=(1050, 225), size=(100, 100), event_code='switch-hint', color=(50, 50, 200)),
            Button(self.win, origin=(1350, 225), size=(100, 100), event_code='step-uptodate', color=(50, 200, 200)),
        ]
        for c in self.components:
            c.init_event(self)

    def register(self, event_type, callback):
        if str(event_type) in self.callbacks:
           self.callbacks[str(event_type)].append(callback)
        else:
            self.callbacks[str(event_type)] = [callback]

    def fetch_input(self):
        recv_queue = self.uisock.recv()
        if recv_queue:
            self.inputs.append(recv_queue)
            # print(f"recv_queue / self.inputs length -> {len(recv_queue)} / {len(self.inputs)}")

    def process_events(self):
        for event in pygame.event.get():
            # print(event.type, pygame.event.event_name(event.type))
            if event.type == pygame.QUIT:
                self.UI_quit()

            if str(event.type) not in self.callbacks:
                continue
            for callback in self.callbacks[str(event.type)]:
                response = callback(event)
                if response:
                    self.inputs.append(response)

    def process_inputs(self):
        tmp_idx_snapshot = self.current_snapshot_idx

        for input in self.inputs:
            # print(f"input (type={type(input)}):\n{input}\n")

            code = input['code']

            if code == 'request-player-action':
                # print(f"-> UI Recv request-player-action")
                self.request_player_action = True

            elif code == 'game-snapshot':

                self.game_snapshots.append(input['data'])
                print(f"New snapshot receive, pause={self.pause}\t, dtime={input['data']['ss_data'].get('dtime', '_')}")

                if not self.pause and self.current_snapshot_idx < len(self.game_snapshots) - 1:
                    self.current_snapshot_idx += 1

            elif code == 'end-game':
                self.uisock.connected = False
                print(f"UIManager: Deconnection asked by GomokuGUIRunner.")
                time.sleep(1)

            elif code == 'board-click':
                x, y = input['data']
                self.board_clicked_action = (x, y)
                print(self.board_clicked_action, self.request_player_action)

            elif code == 'pause-play':
                self.pause = not self.pause
                print(f"Pause={self.pause}")

            elif code == 'step-back' and self.current_snapshot_idx > 0:
                self.current_snapshot_idx -= 1

            elif code == 'step-front' and self.current_snapshot_idx < len(self.game_snapshots) - 1:
                self.current_snapshot_idx += 1

            elif code == 'step-uptodate':
                self.current_snapshot_idx = len(self.game_snapshots) - 1

            elif code == 'switch-hint':
                self.main_board.switch_hint()

        if tmp_idx_snapshot != self.current_snapshot_idx:
            self.snapshot_idx_modified = True

    def update(self):

        if self.snapshot_idx_modified:
            Snapshot.update_from_snapshot(
                self.engine,
                self.game_snapshots[self.current_snapshot_idx]['snapshot'])  # Update local engine to draw
            self.snapshot_idx_modified = False

        if self.request_player_action and self.board_clicked_action and not self.pause:
            print(f"Player action catch")
            if self.engine.is_valid_action(self.board_clicked_action):
                print(f"Player action valid !")

                if self.current_snapshot_idx != len(self.game_snapshots) - 1:  # New state never seen
                    breakpoint() # Need debug ?
                    self.uisock.add_sending_queue({  # Update GUI engine to re-continue with new state
                        'code': 'game-snapshot',
                        'data': self.game_snapshots[self.current_snapshot_idx]['snapshot']
                    })
                    del self.game_snapshots[self.current_snapshot_idx + 1:]  # Remove future snapshots

                self.request_player_action = False
                self.uisock.add_sending_queue({
                    'code': 'response-player-action',
                    'data': self.board_clicked_action,
                })
                # print(f"-> UI Send response-player-action")

        if len(self.game_snapshots):
            ss = self.game_snapshots[self.current_snapshot_idx]
            ss_data = ss['ss_data']
            tottime = ss.get('tottime', ss['time'] - self.init_time)

            for o in self.components:
                o.draw(ss_data=ss_data, ss_i=self.current_snapshot_idx, ss_num=len(self.game_snapshots), tottime=tottime)

        pygame.display.flip()

        self.board_clicked_action = None
        self.inputs = []

    def UI_quit(self):
        # pygame.quit()
        self.uisock.add_sending_queue({
            'code': 'shutdown',
        })
        self.uisock.send_all()
        self.uisock.disconnect()
        self.cross_shutdown = True
        print(f"UIManager: DISCONNECTION.\n")
