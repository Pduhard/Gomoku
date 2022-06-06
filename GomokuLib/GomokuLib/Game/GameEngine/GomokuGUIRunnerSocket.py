from __future__ import annotations
from typing import TYPE_CHECKING, Union

import multiprocessing as mp
from multiprocessing.dummy import Process

import GomokuLib
from GomokuLib.Sockets.UISocketServer import UISocketServer
from GomokuLib.Game.UI.UIManagerSocket import UIManagerSocket
from GomokuLib.Game.GameEngine.Snapshot import Snapshot
from GomokuLib.Game.GameEngine.GomokuRunner import GomokuRunner

import time
from time import perf_counter


class GomokuGUIRunnerSocket(GomokuRunner):

    def __init__(self, start_UI: bool = False, host: str = None, port: int = None,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        print(f"GomokuGUIRunnerSocket.__init__() start UISocketServer server, ")
        self.uisock = UISocketServer(host=host, port=port, name="Runner")

        if start_UI:
            print(f"GomokuGUIRunnerSocket.__init__() start UIManagerSocket client")
            self.gui = UIManagerSocket(
                engine=GomokuLib.Game.GameEngine.Gomoku(),
                win_size=(1500, 1000),
                host=host,
                port=port
            )
            self.gui_proc = Process(target=self.gui)
            self.gui_proc.start()

        self.player_action = None
        self.socket_queue = []

        print("END __init__() GomokuGUIRunnerSocket\n")

    def update_UI(self, **kwargs):
        """
            All kwargs information will be sent to UIManager with new snapshot
        """
        print(f"New snapshot add to socket queue")

        self.uisock.add_sending_queue({
            'code': 'game-snapshot',
            'data': {
                'time': time.time(),
                'snapshot': Snapshot.create_snapshot(self.engine),
                'ss_data': kwargs
            },
        })

    def get_game_data(self, mode: str, dtime_turn):
        return {
            'mode': mode,
            'p1': str(self.players[0]),
            'p2': str(self.players[1]),
            'human_turn': not self.is_bots[self.engine.player_idx ^ 1],
            'turn': self.engine.turn,
            'dtime': dtime_turn, 
            'board': self.engine.board,
            'player_idx': self.engine.player_idx,
            'captures': self.engine.get_captures(),
            'winner': self.engine.winner,
        }

    def _run(self, mode: str = "GomokuGUIRunner.run()", send_all_ss: bool = True):

        self.is_bots = [isinstance(p, GomokuLib.Player.Bot) for p in self.players]
        self.update_UI(
            **self.get_game_data(mode, 0)
        )

        while not self.engine.isover():

            print(f"\nTurn {self.engine.turn}. Player {self.engine.player_idx} to play ...")
            self.UIManager_exchanges()

            p = self.players[self.engine.player_idx]
            is_bot = self.is_bots[self.engine.player_idx]
            time_before_turn = perf_counter()

            player_action = p.play_turn(self)

            time_after_turn = perf_counter()
            dtime_turn = int((time_after_turn - time_before_turn) * 1000)
            print(f"Played in {dtime_turn} ms")

            algo_data = {}
            if is_bot:
                algo_data.update(dict(p.algo.get_state_data(self.engine)))

            self.engine.apply_action(player_action)
            self.engine._next_turn_rules()

            if is_bot:
                algo_data.update(dict(p.algo.get_state_data_after_action(self.engine)))

            # Game data fetching needs to be after apply_action/update_rules and before next_turn
            game_data = self.get_game_data(mode, dtime_turn)

            self.engine._shift_board()

            # Snapshot creation needs to be after next_turn
            self.update_UI(
                **game_data,
                **algo_data
            )

        print(f"Player {self.engine.winner} win.")
        self.GUI_quit(send_all_ss)

    def run(self, *args, **kwargs):
        try:
            super().run(*args, **kwargs)
        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt !")
            self.GUI_quit(False)

        except Exception as e:
            print(f"\nException !!!\n{e}\nClose properly ...")
            self.GUI_quit(False)

    def UIManager_exchanges(self):

        self.uisock.send_all()
        inpt = self.uisock.recv()
        if inpt:
            if inpt['code'] == 'response-player-action':
                ar, ac = inpt['data']
                self.player_action = (ar, ac)

            elif inpt['code'] == 'shutdown':
                print(f"Shutdown by UIManager.")
                self.GUI_quit(False)
                exit(0)

            elif inpt['code'] == 'game-snapshot':
                Snapshot.update_from_snapshot(self.engine, inpt['data'])

    def wait_player_action(self):
        ts = 0
        while True:
            if time.time() > ts + 10:
                self.uisock.add_sending_queue({
                    'code': 'request-player-action'
                })
                print(f"GUI send request-player-action ->")
                ts = time.time()
                
            self.UIManager_exchanges()

            if self.player_action:
                action = self.player_action
                self.player_action = None
                return action

    def GUI_quit(self, send_all_ss):
        print(f"\nGomokuGUIRunner: Send disconnection order to UIManager.")
        self.uisock.add_sending_queue({
            'code': 'end-game'
        })
        self.uisock.send_all(force=send_all_ss)
        self.uisock.disconnect()
        print(f"GomokuGUIRunner: DISCONNECTION.\n")
