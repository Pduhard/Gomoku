from multiprocessing.connection import wait
import threading
import time
import numpy as np

from numba import njit
from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Game.GameEngine.Snapshot import Snapshot
from GomokuLib.Algo.MCTSNjit import MCTSNjit


# @njit()
# def do_n_iter(mcts, iter):
#     for i in range(iter):
#         mcts.do_n_iter()


class HumanHints:

    def __init__(self, game_engine: Gomoku, batch_iter: int = 100) -> None:
        self.engine = game_engine.clone()
        self.is_running = False
        self.batch_iter = batch_iter

        self.mcts = MCTSNjit(
            engine=self.engine,
            iter=self.batch_iter,
            pruning=True,
            rollingout_turns=5,
            with_new_heuristic=True
        )
        print(f"HumanHints: Numba compilation of MCTSNjit start ...")
        ts = time.time()
        self.mcts.do_n_iter(self.engine, 1)
        print(f"HumanHints: Numba compilation of MCTSNjit is finished.\tdtime: {time.time() - ts}")

        self.thread = threading.Thread(
            target=self.compute_hints
        )
        self.lock = threading.Lock()
        # print(f"threading.main_thread: {threading.main_thread()}")

    def update_from_snapshot(self, snapshot: Snapshot):
        Snapshot.update_from_snapshot(
            self.engine,
            snapshot
        )

    def fetch_hints(self) -> tuple[np.ndarray, np.ndarray]:
        print(f"HumanHints: fetch_hints ...")
        # print(f"Threads {threading.current_thread()}\tin\t{threading.enumerate()}")

        if not self.thread.is_alive():
            self.is_running = True
            self.thread.start()

        print(f"HumanHints: Thread running ...")
        self.lock.acquire()
        mcts_data = self.mcts.get_state_data(self.engine)
        self.lock.release()
        print(f"HumanHints: Succesfully fetch data")
        return mcts_data

    def stop(self):
        if self.thread.is_alive():
            print(f"HumanHints: Thread stop order\t{self.thread._target}")
            self.is_running = False
            self.thread.join()
            self.thread = threading.Thread(
                target=self.compute_hints
            )
            print(f"HumanHints: Thread joined\t{self.thread._target}")

    def compute_hints(self):
        print(f"HumanHints: Start Thread for {self.batch_iter} mcts iterations")
        # print(f"Threads {threading.current_thread()}\tin\t{threading.enumerate()}")
        while self.is_running:
            self.lock.acquire()
            print(f"HumanHints: do_n_iter: {self.batch_iter}")
            self.mcts.do_n_iter(self.engine, self.batch_iter)
            self.lock.release()
            time.sleep(0.2)
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
