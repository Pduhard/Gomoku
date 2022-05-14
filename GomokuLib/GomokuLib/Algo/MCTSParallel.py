import concurrent.futures

import numpy as np
from numba import njit, prange

from ..Game.GameEngine import Gomoku

from multiprocessing import cpu_count

from GomokuLib.Algo.MCTSEvalLazy import MCTSEvalLazy
from GomokuLib.Algo.MCTSWorker import MCTSWorker
#
# @njit(parallel=True, nogil=True)
# # @njit(parallel=True)
# # @njit(nogil=True)
# # @njit()
# def _call_worker():
#
#     a = np.empty((1000, 1000), dtype=np.float32)
#     for i in prange(1000):
#         for j in prange(1000):
#             a[i, j] = np.sqrt(7)
#     return a
#
# @njit(nogil=True)
# @njit()
# def call_worker():
#     return _call_worker()

class MCTSParallel(MCTSEvalLazy):
# class MCTSParallel:

    def __init__(self,
                 num_workers: int = 3,
                 batch_size: int = 1,
                 *args, **kwargs
                 ) -> None:
        # super().__init__(*args, **kwargs)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.workers_data = []
        self.workers = [
            MCTSWorker(id=i, *args, **kwargs)
            for i in range(self.num_workers)
        ]
        self.pool = concurrent.futures.ThreadPoolExecutor(
        # self.pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=None,
            # thread_name_prefix="test"
        )
        print("Create pool successfully")

    def __str__(self):
        return f"MCTSParallel with {self.num_workers} workers and  iterations"

    def __call__(self, game_engine: Gomoku) -> tuple:
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
        # print(f"Worker {res[1]} response has been receive: {res[2:]}")
        self.workers_data.append(res)

    def update_state_data(self, worker: MCTSWorker):
        # print("Update state_data.")
        self.workers_data = []
        return [worker]
