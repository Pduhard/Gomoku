from __future__ import annotations
from typing import TYPE_CHECKING, Union

import multiprocessing as mp
from multiprocessing.dummy import Process

import GomokuLib
from GomokuLib.Sockets.UISocket import UISocket
from GomokuLib.Game.UI.UIManagerSocket import UIManagerSocket
from GomokuLib.Game.GameEngine.Snapshot import Snapshot
from GomokuLib.Game.GameEngine.GomokuRunner import GomokuRunner

import time


class GomokuGUIRunnerSocket(GomokuRunner):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.gui_outqueue = mp.Queue()
        self.gui_inqueue = mp.Queue()

        # self.engine is defined in super
        self.gui = UIManagerSocket(self.engine, (1500, 1000))
        self.gui_proc = Process(target=self.gui, args=(self.gui_outqueue, self.gui_inqueue))
        self.gui_proc.start()

        print(f"UISocket GUI start init")
        self.uisock = UISocket(as_server=True, name="Runner")
        self.uisock.start_sock_thread()
        print(f"UISocket GUI end init")

        self.player_action = None
        self.socket_queue = []

        print("END __init__() GomokuGUIRunner\n")

    def update_UI(self, **kwargs):
        """
            All kwargs information will be sent to UIManager with new snapshot
        """
        print(f"New snapshot add to socket queue")

        # self.gui_outqueue.put({
        #     'code': 'game-snapshot',
        #     'data': {
        #         'time': time.time(),
        #         'snapshot': Snapshot.create_snapshot(self.engine),
        #         'ss_data': kwargs
        #     },
        # })
        self.uisock.add_sending_queue({
            'code': 'game-snapshot',
            'data': {
                'time': time.time(),
                'snapshot': Snapshot.create_snapshot(self.engine),
                'ss_data': kwargs
            },
        })

    def _run(self, players, mode: str = "GomokuGUIRunner.run()"):

        while not self.engine.isover():
            self.get_gui_input()

            p = players[self.engine.player_idx]
            player_action = p.play_turn(engine=self.engine, runner=self)

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
        time.sleep(5)
        # self.uisock.stop_sock_thread()
        # self.uisock.disconnect()

    # def get_gui_input(self):
    #
    #     try:
    #         while True:
    #             inpt = self.gui_inqueue.get_nowait()  # raise Empty Execption
    #
    #             if inpt['code'] == 'response-player-action':
    #                 ar, ac = inpt['data']
    #                 self.player_action = (ar, ac)
    #
    #             elif inpt['code'] == 'shutdown':
    #                 exit(0)
    #
    #             elif inpt['code'] == 'game-snapshot':
    #                 breakpoint()
    #                 Snapshot.update_from_snapshot(self.engine, inpt['data'])
    #     except:
    #         pass
    def get_gui_input(self):

        for inpt in self.uisock.get_recv_queue():

            if inpt['code'] == 'response-player-action':
                ar, ac = inpt['data']
                self.player_action = (ar, ac)

            elif inpt['code'] == 'shutdown':
                self.GUI_quit()

            elif inpt['code'] == 'game-snapshot':
                breakpoint() # Need debug ?
                Snapshot.update_from_snapshot(self.engine, inpt['data'])

    def wait_player_action(self):
        # self.gui_outqueue.put({
        #     'code': 'request-player-action'
        # })
        self.uisock.add_sending_queue({
            'code': 'request-player-action'
        })
        print(f"<- GUI send request-player-action ...")
        while True:
            self.get_gui_input()

            if self.player_action:
                action = self.player_action
                self.player_action = None
                return action

    def GUI_quit(self):
        self.uisock.stop_sock_thread()
        self.uisock.disconnect()
        exit(0)
