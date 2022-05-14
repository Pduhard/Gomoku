import concurrent.futures

import numpy as np
from numba import njit, prange
from numba.experimental import jitclass

from GomokuLib.Algo.MCTSLazy import MCTSLazy

from ..Game.GameEngine import Gomoku

from multiprocessing import cpu_count

from GomokuLib.Algo.MCTSEvalLazy import MCTSEvalLazy
from GomokuLib.Algo.MCTSWorker import MCTSWorker, GomokuJit
from GomokuLib.Algo.MCTS import MCTS


class MCTSParallel(MCTSLazy):

    def __init__(self,
                 engine: Gomoku,
                 num_workers: int = 1,
                 batch_size: int = 4,
                 *args, **kwargs
                 ) -> None:
        # super().__init__(engine, *args, **kwargs)

        self.engine = engine.clone()
        res = MCTSWorker(self.engine, np.int32(0)).do_your_fck_work(self.engine, None, np.int32(0))
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.workers_data = []
        self.workers = [
            # MCTSWorker(self.engine, i)
            MCTSWorker(GomokuJit(), np.int32(i))
            for i in range(self.num_workers)
        ]
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=None,
        )
        print(f"Successfully create {len(self.workers)} workers")

    def __str__(self):
        return f"MCTSParallel with {self.num_workers} workers and  iterations"

    def __call__(self) -> tuple:
        print(f"\n[MCTSParallel begin __call__()] -> {self.num_workers} workers for 100 iter\n")

        # Submit all Workers for an iteration
        futures = [
            # self.pool.submit(call_worker)
            self.pool.submit(worker)
            for worker in self.workers
        ]

        # Set callbacks functions to handle Workers responses
        for future in futures:
            future.add_done_callback(self.update_workers_return)

        # Main loop
        not_done_futures = futures
        _iter = 0
        while _iter < 1000:

            # Wait until we get for the first Worker response
            done_future, not_done_futures = map(list, concurrent.futures.wait(not_done_futures, return_when=concurrent.futures.FIRST_COMPLETED))

            for future in done_future:
                if len(self.workers_data) >= self.batch_size:
                    # Submit a Thread to update state_data
                    f = self.pool.submit(self.update_state_data, future.result()[0])

                else:
                    # Submit a Worker for a new iteration
                    # f = self.pool.submit(call_worker)
                    f = self.pool.submit(future.result()[0])
                    f.add_done_callback(self.update_workers_return)
                    _iter += 1
                not_done_futures.append(f)

        # Wait until all workers has finished
        concurrent.futures.wait(futures)
        self.update_state_data(None)

        print(f"\n[MCTSParallel end __call__()] -> {self.num_workers} workers for 100 iter\n")

    def update_workers_return(self, future):
        res = future.result()
        print(f"Worker {res[1]} response has been receive: {res[2:]}")
        self.workers_data.append(res)

    def update_state_data(self, worker: MCTSWorker):
        print("Update state_data.")
        self.workers_data = []
        return [worker]
