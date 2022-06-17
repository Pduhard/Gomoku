import threading
import time
import numpy as np

from GomokuLib.Algo.MCTSNjit import MCTSNjit
from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Game.GameEngine.Snapshot import Snapshot


class HumanHints:

    def __init__(self, game_engine: Gomoku,
                 max_iter: int = 25000
                 ) -> None:
        print(f"\nHumanHints: __init__(): START")

        self.engine = game_engine.clone()
        self.max_state_iter = max_iter
        self.is_running = False

        print(f"\nHumanHints: MCTSNjit: Numba compilation starting ...")
        print(f"HumanHints: MCTSNjit: __init__(): START")
        ts = time.time()
        self.mcts = MCTSNjit(
            engine=self.engine,
            iter=0,
            time=100,
        )
        print(f"HumanHints: MCTSNjit: __init__(): DONE")

        # self.mcts.compile(self.engine)
        print(f"HumanHints: MCTSNjit: Numba compilation is finished (dtime={round(time.time() - ts, 1)})\n")

        self.thread = threading.Thread(
            target=self.compute_hints
        )

        print(f"\nHumanHints: __init__(): DONE")

    def update_from_snapshot(self, snapshot: Snapshot):
        Snapshot.update_from_snapshot(
            self.engine,
            snapshot
        )

    def start(self):
        if not self.thread.is_alive():
            self.is_running = True
            self.thread = threading.Thread(
                target=self.compute_hints
            )
            self.thread.start()

    def stop(self):
        if self.thread.is_alive():
            self.is_running = False
            self.thread.join()
            print(f"HumanHints: Thread joined.")

    def fetch_hints(self) -> tuple[np.ndarray, np.ndarray]:

        mcts_data = dict(self.mcts.get_state_data(self.engine))

        try:
            if mcts_data['mcts_state_data'][0]['visits'] > self.max_state_iter:
                # print(f"HumanHints: Automatic stop of MCTSNjit iterations")
                self.stop()
            else:
                self.start()
        except Exception as e:
            print(f"HumanHints: Unable to control automatic stop of iterations:\n\t{e}")

        return mcts_data

    def compute_hints(self):
        print(f"HumanHints: Start Thread.")

        try:
            self.mcts.init()
            while self.is_running:
                # print(f"HumanHints: do_n_iter: {self.batch_iter}")
                self.mcts.do_your_fck_work(self.engine)
                time.sleep(0.1)     # Very, very important. Unless that, UI cannot respond
        except Exception as e:
            print(f"HumanHints: Error while computing Human hints:\n\t{e}")
