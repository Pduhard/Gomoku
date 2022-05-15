import concurrent.futures

import numpy as np
from numba import njit, prange
from numba.experimental import jitclass

from GomokuLib.Algo.MCTSLazy import MCTSLazy

from ..Game.GameEngine import Gomoku

from multiprocessing import cpu_count

from GomokuLib.Algo.MCTSEvalLazy import MCTSEvalLazy
from GomokuLib.Algo.MCTSWorker import MCTSWorker, GomokuJit, path_nb_dtype
from GomokuLib.Algo.MCTS import MCTS

import numba as nb


class MCTSParallel(MCTSLazy):

    def __init__(self,
                 engine: Gomoku,
                 num_workers: int = 1,
                 batch_size: int = 10,
                 *args, **kwargs
                 ) -> None:
        # super().__init__(engine, *args, **kwargs)

        self.engine = engine.clone()
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.states = {}
        self.workers_states_buff = {}
        # self.workers_states_data_buff = np.zeros(self.batch_size, dtype=state_data_dtype)
        # self.workers_states_data_len = 0

        # breakpoint()
        # self.workers = [
        #     MCTSWorker(
        #         GomokuJit(),
        #         np.int32(i)
        #     )
        #     for i in range(self.num_workers)
        # ]
        # self.pool = concurrent.futures.ThreadPoolExecutor(
        #     max_workers=None,
        # )
        # print(f"Successfully create {len(self.workers)} workers")

    def __str__(self):
        return f"MCTSParallel with {self.num_workers} workers and  iterations"

    def __call__(self) -> tuple:
        print(f"\n[MCTSParallel begin __call__()] -> {self.num_workers} workers for 100 iter\n")

        # Submit all Workers for an iteration
        futures = [
            # self.pool.submit(call_worker)
            self.pool.submit(worker.do_your_fck_work)
            for worker in self.workers
        ]

        # Set callbacks functions to handle Workers responses
        for future in futures:
            future.add_done_callback(self.update_workers_return)

        # Main loop
        not_done_futures = futures
        _iter = 0
        while _iter < 10:

            # Wait until we get for the first Worker response
            done_future, not_done_futures = map(
                list,
                concurrent.futures.wait(
                    not_done_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
            )

            for future in done_future:
                if len(self.workers_data) >= self.batch_size:
                    # Submit a Thread to update state_data
                    f = self.pool.submit(self.update_state_data, future.result())

                else:
                    # Submit a Worker for a new iteration
                    # f = self.pool.submit(call_worker)
                    f = self.pool.submit(future.result())
                    f.add_done_callback(self.update_workers_return)
                    _iter += 1
                not_done_futures.append(f)

        # Wait until all workers has finished
        concurrent.futures.wait(futures)
        self.update_state_data(None)

        print(f"\n[MCTSParallel end __call__()] -> {self.num_workers} workers for 100 iter\n")

    def update_workers_return(self, future):
        worker_id = future.result()
        print(f"Worker {worker_id}'s response has been receive.")
        print(f"workers_state_data_buff: {self.workers_state_data_buff}")

        self.workers_states_buff.append(self.workers_state_data_buff[worker_id])

        print(f"self.statehash_buff[worker_id]: {self.workers_statehash_buff[worker_id]}")
        print(f"self.workers_p_idx_buff[worker_id]: {self.workers_p_idx_buff[worker_id]}")
        print(f"self.workers_bestaction_buff[worker_id]: {self.workers_bestaction_buff[worker_id]}")

    def update_state_data(self, worker_id: int):
        print(f"Update state_data. Thread {worker_id}")

        # model prediction
        # backprop

        self.states.append(self.workers_states_buff)
        self.workers_states_buff = np.zeros(1, dtype=state_data_dtype)

        # Return worker reference to start a new iteration with this thread
        return self.workers[worker_id].do_your_fck_work


    def backpropagation(self, path: list):

        reward = self.reward
        for mem in path[::-1]:
            self.backprop_memory(mem, reward)
            reward = 1 - reward

    def backprop_memory(self, memory: tuple, reward: float):
        _, statehash, bestaction = memory

        state_data = self.states[statehash]

        state_data['Visits'] += 1                           # update n count
        state_data['Rewards'] += reward                     # update state value
        if bestaction is None:
            return

        r, c = bestaction.action
        state_data['StateAction'][..., r, c] += [1, reward]  # update state-action count / value


    def test(self):

        worker = MCTSWorker(
            GomokuJit(),
            np.int32(0)
        )

        worker_ret = worker.do_your_fck_work()
        self.workers_states_buff[worker_ret['path'][-1]['statehash']] = worker_ret['leaf_data'].copy()
        # self.workers_states_data_len += 1

        print("1\n", self.workers_states_buff)
        
        worker_ret = worker.do_your_fck_work()
        self.workers_states_buff[worker_ret['path'][-1]['statehash']] = worker_ret['leaf_data'].copy()

        print("2\n", self.workers_states_buff)

        self.states.update(self.workers_states_buff)

        print("3\n", self.states)
