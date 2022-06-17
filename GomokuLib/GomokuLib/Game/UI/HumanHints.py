from multiprocessing.connection import wait
import threading
import time
import numpy as np

from numba import njit
from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Game.GameEngine.Snapshot import Snapshot
from GomokuLib.Algo.MCTSNjit import MCTSNjit


class HumanHints:

    def __init__(self, game_engine: Gomoku,
                 batch_iter: int = 1000,
                 max_iter: int = 42000
                 ) -> None:
        print(f"\nHumanHints: __init__(): START")

        self.engine = game_engine.clone()
        self.batch_iter = batch_iter
        self.max_iter = max_iter
        self.is_running = False

        self.mcts = MCTSNjit(
            engine=self.engine,
            iter=self.batch_iter
        )
        print(f"\nHumanHints: MCTSNjit: Numba compilation starting ...")
        ts = time.time()
        self.mcts.compile(self.engine)
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
            self.thread.start()

    def stop(self):
        if self.thread.is_alive():
            self.is_running = False
            self.thread.join()
            self.thread = threading.Thread(
                target=self.compute_hints
            )
            print(f"HumanHints: Thread joined.")

    def fetch_hints(self) -> tuple[np.ndarray, np.ndarray]:

        mcts_data = dict(self.mcts.get_state_data(self.engine))

        try:
            if mcts_data['mcts_state_data'][0]['visits'] >= self.max_iter:
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
            while self.is_running:
                # print(f"HumanHints: do_n_iter: {self.batch_iter}")
                self.mcts.do_your_fck_work(self.engine, self.batch_iter, 0)
                time.sleep(0.1)
        except Exception as e:
            print(f"HumanHints: Error while computing Human hints:\n\t{e}")


if __name__ == "__main__":
    engine = Gomoku()
    hh = HumanHints(engine, 50)

    hints = hh.fetch_hints()    
    print(f"First hints: ", hints.dtype)
    # print(f"threading.main_thread: {threading.main_thread()}")

    i = 0
    ts = time.time()
    while time.time() < ts + 3:
        hints = hh.fetch_hints()    
        print(f"Hints {i}:\t", hints.dtype)
        time.sleep(0.5)
        i += 1

    hh.stop()
