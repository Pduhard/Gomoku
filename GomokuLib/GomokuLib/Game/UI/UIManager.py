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
from GomokuLib.Game.UI.HumanHints import HumanHints

class UIManager:

    def __init__(self, win_size: tuple, host: str = None, port: int = None):

        assert len(win_size) == 2
        self.ox, self.oy = 0, 0
        self.dx, self.dy = win_size
        self.host = host
        self.port = port

        self.engine = Gomoku()
        self.board_size = self.engine.board_size
        self.humanHints = HumanHints(self.engine)

    def __call__(self): # Thread function

        print("UIManager __call__()\n")
        self.init()

        self.cross_shutdown = False
        while not self.cross_shutdown:

            self.fetch_input()
            self.process_events()
            self.process_inputs()

            self.update_engines()
            self.update_components()
            self.handle_human_click()

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

        self.initUI()
        
    def initUI(self):

        print(f"board_size: {self.board_size}")

        print("init GUI")
        pygame.init()

        # self.win = pygame.display.set_mode((self.dx, self.dy), pygame.RESIZABLE)
        self.win = pygame.display.set_mode((self.dx, self.dy))
        # self.win.convert_alpha()

        button_prop = 0.1       # Proportion size of buttons
        bgrid_begin_x = 0.68    # Start x of button grid
        bgrid_begin_y = 0.05    # Start y of button grid

        # Reperes/Rules in the windows to create all components
        prop_x = [0.0, 0.66, bgrid_begin_x, bgrid_begin_x + button_prop, bgrid_begin_x + 2 * button_prop, 1.0]
        prop_y = [0.0, bgrid_begin_y, bgrid_begin_y + button_prop, bgrid_begin_y + 2 * button_prop, 0.40, 1.0]
        self.rules = {
            'x': [self.ox + p * self.dx for p in prop_x],
            'y': [self.oy + p * self.dy for p in prop_y]
        }

        board_square_size = min(self.rules['x'][1], self.rules['y'][-1])
        self.main_board = Board(
            self.win,
            origin=(self.rules['x'][0], self.rules['y'][0]),
            size=(board_square_size, board_square_size),
            board_size=self.board_size,
            humanHints=self.humanHints
        )
        self.display = Display(
            self.win,
            origin=(self.rules['x'][2], self.rules['y'][-2]),
            size=(
                self.rules['x'][-1] - self.rules['x'][2],
                self.rules['y'][-1] - self.rules['y'][-2]
            )
        )

        button_size = button_prop * self.dx, button_prop * self.dy
        button_data = []
        for pyi in range(1, 4):
            for pxi in range(2, 5):
                button_data.append({
                    'win': self.win,
                    'origin': (self.rules['x'][pxi], self.rules['y'][pyi]),
                    'size': button_size
                })

        self.components = [
            self.main_board,
            self.display,

            Button(**button_data[0], event_code='step-back', color=(0, 255, 255)),
            Button(**button_data[1], event_code='pause-play', color=(50, 200, 50), num_states=2),
            Button(**button_data[2], event_code='step-front', color=(0, 255, 255)),

            Button(**button_data[3], event_code='switch-hint', color=(100, 100, 200), num_states=4),
            Button(**button_data[4], event_code='human-hints', color=(100, 100, 200), num_states=2),
            Button(**button_data[5], event_code='step-uptodate', color=(0, 255, 255)),

            Button(**button_data[6], event_code='new-game', color=(50, 200, 50)),
            Button(**button_data[7], event_code='debug-mode', color=(50, 200, 200), num_states=2),
            Button(**button_data[8], event_code='send-snapshot', color=(50, 200, 200)),
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
            self.inputs = [recv_queue]
        else:
            self.inputs = []
            # print(f"recv_queue / self.inputs length -> {len(recv_queue)} / {len(self.inputs)}")

    def process_events(self):
        for event in pygame.event.get():
            # print(event.type, pygame.event.event_name(event.type))
            if event.type == pygame.QUIT:
                self.UI_quit()

            # elif event.type == pygame.VIDEORESIZE:    # Need to update all components with new size ...
            #     self.dx, self.dy = event.w, event.h

            elif str(event.type) in self.callbacks:
                for callback in self.callbacks[str(event.type)]:
                    response = callback(event)
                    if response:
                        self.inputs.append(response)

    def process_inputs(self):

        tmp_idx_snapshot = self.current_snapshot_idx
        self.board_clicked_action = None

        for input in self.inputs:
            # print(f"input (type={type(input)}):\n{input}\n")

            code = input['code']

            if code == 'request-player-action':
                print(f"-> UI Recv request-player-action")
                self.request_player_action = True

            elif code == 'game-snapshot':

                self.game_snapshots.append(input['data'])
                print(f"New snapshot receive, pause={self.pause}\t, dtime={input['data']['ss_data'].get('dtime', '_')}")

                if not self.pause and self.current_snapshot_idx < len(self.game_snapshots) - 1:
                    self.current_snapshot_idx += 1

            elif code == 'board-click':
                x, y = input['data']
                self.board_clicked_action = (x, y)
                print(f"Request Human: {self.request_player_action} | Receive action: {self.board_clicked_action}")

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
                self.main_board.switch_hint(input['state'])

            elif code == 'end-game':
                self.uisock.connected = False
                self.humanHints.stop()
                print(f"UIManager: Deconnection asked by GomokuGUIRunner.")
                time.sleep(1)   # Very important and we will never talk about why ... Please.

            elif code == 'new-game':
                self.uisock.add_sending_queue({
                    'code': 'new-game'
                })

        if tmp_idx_snapshot != self.current_snapshot_idx:
            self.snapshot_idx_modified = True

    def update_engines(self):

        if self.snapshot_idx_modified:
            snapshot = self.game_snapshots[self.current_snapshot_idx]['snapshot']
            Snapshot.update_from_snapshot(
                self.engine,
                snapshot
            )  # Update local engine to test valid action

            self.humanHints.update_from_snapshot(snapshot)
            self.snapshot_idx_modified = False

    def update_components(self):

        if len(self.game_snapshots):
            ss = self.game_snapshots[self.current_snapshot_idx]
            ss_data = ss['ss_data']
            tottime = ss.get('tottime', ss['time'] - self.init_time)

            for o in self.components:
                o.draw(ss_data=ss_data, snapshot=ss['snapshot'], ss_i=self.current_snapshot_idx, ss_num=len(self.game_snapshots), tottime=tottime)

        pygame.display.flip()

    def handle_human_click(self):

        if self.request_player_action and self.board_clicked_action and not self.pause:
            print(f"Player action catch")
            if self.engine.is_valid_action(self.board_clicked_action):
                print(f"Player action valid !")

                if self.current_snapshot_idx != len(self.game_snapshots) - 1: # New state never seen

                    current_ss = self.game_snapshots[self.current_snapshot_idx]['snapshot']
                    lastest_ss = self.game_snapshots[-1]['snapshot']

                    print(f"current_ss['player_idx'] == lastest_ss['player_idx'] ? {current_ss['player_idx']} == {lastest_ss['player_idx']}")
                    if current_ss['player_idx'] == lastest_ss['player_idx']:    # Human can only play at its turns 
                        # breakpoint() # Need debug ?
                        self.uisock.add_sending_queue({  # Update GUI engine to re-continue with new state
                            'code': 'game-snapshot',
                            'data': current_ss
                        })
                        del self.game_snapshots[self.current_snapshot_idx + 1:]  # Remove future snapshots

                    else:
                        return

                self.humanHints.stop()
                self.request_player_action = False
                self.uisock.add_sending_queue({
                    'code': 'response-player-action',
                    'data': self.board_clicked_action,
                })
                # print(f"-> UI Send response-player-action")
            
            else:
                print(f"Not a valid action ! -> {self.board_clicked_action}")

    def debug_mode():
        """
            while on larrete pas:
                Poser ou enlever une stone
                update les autres composents de l'UI
                Utiliser Humanhints
            update le GomokuGUIRunner
        """
        pass

    def UI_quit(self):
        self.humanHints.stop()
        self.uisock.add_sending_queue({
            'code': 'shutdown',
        })
        self.uisock.send_all()
        self.uisock.disconnect()
        self.cross_shutdown = True
        print(f"UIManager: DISCONNECTION.\n")
        pygame.quit()
        exit(0)

