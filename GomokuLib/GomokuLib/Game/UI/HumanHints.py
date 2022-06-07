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
                 batch_iter: int = 500, max_iter: int = 42420) -> None:
        self.engine = game_engine.clone()
        self.batch_iter = batch_iter
        self.max_iter = max_iter
        self.is_running = False

        self.mcts = MCTSNjit(
            engine=self.engine,
            iter=self.batch_iter,
            pruning=True,
            rollingout_turns=10,
        )
        print(f"HumanHints: Numba compilation of MCTSNjit start ...")
        ts = time.time()
        # self.mcts.do_n_iter(self.engine, 1)
        print(f"HumanHints: Numba compilation of MCTSNjit is finished.\tdtime: {time.time() - ts}")

        self.thread = threading.Thread(
            target=self.compute_hints
        )

    def update_from_snapshot(self, snapshot: Snapshot):
        Snapshot.update_from_snapshot(
            self.engine,
            snapshot
        )

    def start(self):
        if not self.thread.is_alive():
            self.is_running = True
            self.thread.start()
        # print(f"HumanHints: Thread running ...")

    def stop(self):
        if self.thread.is_alive():
            print(f"HumanHints: Thread stop order")
            self.is_running = False
            self.thread.join()
            self.thread = threading.Thread(
                target=self.compute_hints
            )
            print(f"HumanHints: Thread joined")

    def fetch_hints(self) -> tuple[np.ndarray, np.ndarray]:

        mcts_data = dict(self.mcts.get_state_data(self.engine))

        try:
            if mcts_data['mcts_state_data'][0]['visits'] >= self.max_iter:
                self.stop()
            else:
                self.start()
        except:
            print(f"HumanHints: Unable to control automatic stop of iterations")

        # print(f"HumanHints: Succesfully fetch data")
        return mcts_data

    def compute_hints(self):
        print(f"HumanHints: Start Thread for {self.batch_iter} mcts iterations")

        try:
            while self.is_running:
                # print(f"HumanHints: do_n_iter: {self.batch_iter}")
                self.mcts.do_n_iter(self.engine, self.batch_iter)
                time.sleep(0.1)
        except Exception as e:
            print(f"HumanHints: Error while computing Human hints:\n\t{e}")

        print(f"HumanHints: Thread finish")


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
