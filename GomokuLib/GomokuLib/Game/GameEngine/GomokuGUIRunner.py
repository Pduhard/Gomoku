from multiprocessing.dummy import Process

import GomokuLib
from GomokuLib.Sockets.UISocketServer import UISocketServer
from GomokuLib.Game.UI.UIManager import UIManager
from GomokuLib.Game.GameEngine.Snapshot import Snapshot
from GomokuLib.Game.GameEngine.GomokuRunner import GomokuRunner

import time
from time import perf_counter


class UIShutdown(Exception):
    pass


class GomokuGUIRunner(GomokuRunner):

    def __init__(self, start_UI: bool = False, host: str = None, port: int = None,
                 *args, **kwargs) -> None:
        print(f"GomokuGUIRunner: __init__(): START")

        super().__init__(*args, **kwargs)

        self.uisock = UISocketServer(host=host, port=port, name="Runner")

        if start_UI:
            self.gui = UIManager(
                engine=self.engine,
                win_size=(1500, 1000),
                host=host,
                port=port
            )
            self.gui_proc = Process(target=self.gui)
            self.gui_proc.start()

        self.player_action = None
        self.socket_queue = []
        self.init_snapshot = False

        print("GomokuGUIRunner: __init__(): DONE\n")

    def update_UI(self, **kwargs):
        """
            All kwargs information will be sent to UIManager with new snapshot
        """
        self.uisock.add_sending_queue({
            'code': 'game-snapshot',
            'data': {
                'time': time.time(),
                'snapshot': Snapshot.create_snapshot(self.engine),
                'ss_data': kwargs
            },
        })

    def get_game_data(self, dtime_turn):
        return {
            'mode': "GomokuGUIRunner.run()",
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

    def _run(self):

        self.is_bots = [isinstance(p, GomokuLib.Player.Bot) for p in self.players]
        self.update_UI(
            **self.get_game_data(0)
        )

        while not self.engine.isover():

            print(f"\n--- Turn {self.engine.turn}. Player {self.engine.player_idx} is playing ...")
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
            self.engine.next_turn()

            # Snapshot creation needs to be after next_turn
            self.update_UI(
                **algo_data,
                **self.get_game_data(dtime_turn),
            )

    def run(self, *args, **kwargs):

        winners = []
        self.play = True
        try:
            while True:

                if self.play:
                    w = super().run(init_snapshot=self.init_snapshot, *args, **kwargs)
                    winners.append(w)
                    self.play = False
                    self.init_snapshot = None
                    print(f"GomokuGUIRunner: run(): Waiting for a new game ...\n")

                self.UIManager_exchanges()
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\nGomokuGUIRunner: run(): Keyboard interruption !")
        
        except UIShutdown:
            print(f"\nGomokuGUIRunner: run(): Shutdown by UIManager.")
            self.GUI_quit(shutdown_UI=False)
            return winners

        except Exception as e:
            print(f"\nGomokuGUIRunner: run(): Exception:\n\t{e}\n")

        self.GUI_quit()
        return winners

    def UIManager_exchanges(self):

        self.uisock.send_all()
        inpt = self.uisock.recv()
        if inpt:
            if inpt['code'] == 'response-player-action':
                self.player_action = inpt['data']

            elif inpt['code'] == 'shutdown':
                raise UIShutdown()

            elif inpt['code'] == 'game-snapshot':
                if self.play:
                    Snapshot.update_from_snapshot(self.engine, inpt['data'])
                else:
                    self.init_snapshot = inpt['data']
                self.play = True

            elif inpt['code'] == 'new-game':
                self.play = True

    def wait_player_action(self):
        ts = 0
        while True:

            self.UIManager_exchanges()

            if self.player_action:
                action = self.player_action
                self.player_action = None
                return action

            # Make sure UIManager receive the request
            elif time.time() > ts + 10:
                self.uisock.add_sending_queue({
                    'code': 'request-player-action'
                })
                print(f"\nGomokuGUIRunner: Send 'request-player-action' order")
                ts = time.time()

    def GUI_quit(self, shutdown_UI: bool = True):
        if shutdown_UI:
            print(f"\nGomokuGUIRunner: Send disconnection order to UIManager.")
            self.uisock.add_sending_queue({
                'code': 'end-game'
            })

        self.uisock.send_all()
        self.uisock.disconnect()
        print(f"GomokuGUIRunner: DISCONNECTION.\n")
