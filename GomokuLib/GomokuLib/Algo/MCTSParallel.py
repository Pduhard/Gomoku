import concurrent.futures
import threading

import numpy as np

import numba as nb
from numba import njit, prange
from numba.experimental import jitclass

import GomokuLib.Typing as Typing

from GomokuLib.Algo.MCTS import MCTS
from GomokuLib.Algo.MCTSLazy import MCTSLazy
from GomokuLib.Algo.MCTSWorker import MCTSWorker
from GomokuLib.Algo.MCTSEvalLazy import MCTSEvalLazy
from GomokuLib.Game.GameEngine import Gomoku



class MCTSParallel(MCTSLazy):

    def __init__(self,
                 engine: Gomoku,
                 num_workers: int = 3,
                 batch_size: int = 5,
                 mcts_iter: int = 10,
                 *args, **kwargs
                 ) -> None:
        # super().__init__(engine, *args, **kwargs)

        self.engine = engine.clone()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mcts_iter = mcts_iter

        self.states = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        self.states_buff = np.recarray(shape=(self.batch_size,), dtype=Typing.StateDataDtype)
        self.path_buff = np.recarray(shape=(self.batch_size,), dtype=Typing.PathDtype)
        self._buff_id = 0
        self.buff_lock = threading.Lock()
        print(f"Parallel __init__() shapes: {self.states_buff.shape}\t {self.path_buff.shape}")

        self.workers_state_data_buff = np.recarray(shape=(self.num_workers,), dtype=Typing.StateDataDtype)
        self.workers_path_buff = np.recarray(shape=(self.num_workers,), dtype=Typing.PathDtype)
        print(f"Parallel __init__() shapes: {self.workers_state_data_buff.shape}\t {self.workers_path_buff.shape}")

        engine = Gomoku()
        self.states['12'] = np.recarray(1, dtype=Typing.StateDataDtype)
        self.states['12'][0].Actions[-1, -1] = 42
        # breakpoint()
        self.workers = [
            MCTSWorker(
                np.int32(i),
                engine,
                self.workers_state_data_buff,
                self.workers_path_buff,
                self.states
            )
            for i in range(self.num_workers)
        ]
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=None,
        )
        print(f"Parallel: Successfully create {len(self.workers)} workers")

    def __str__(self):
        return f"MCTSParallel with {self.num_workers} workers and  iterations"

    def __call__(self) -> tuple:
        print(f"\n[MCTSParallel begin __call__()] -> {self.num_workers} workers for {self.mcts_iter} iter\n")

        # Submit all Workers for an iteration
        futures = [
            # self.pool.submit(call_worker)
            self.pool.submit(worker.do_your_fck_work)
            for worker in self.workers
        ]

        # Set callbacks functions to handle Workers responses
        for future in futures:
            future.add_done_callback(self.handle_workers_return)

        # Main loop
        not_done_futures = futures
        _iter = len(futures)
        while _iter < self.mcts_iter:

            # Wait until we get for the first Worker response
            done_future, not_done_futures = map(
                list,
                concurrent.futures.wait(
                    not_done_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
            )

            for future in done_future:
                # Submit done Workers for a new iteration
                f = self.pool.submit(self.workers[future.result()].do_your_fck_work)
                f.add_done_callback(self.handle_workers_return)
                not_done_futures.append(f)
                _iter += 1

        # Wait until all workers has finished
        concurrent.futures.wait(futures)
        self.update_states()

        print(f"\n[MCTSParallel end __call__()] -> {self.num_workers} workers for 100 iter\n")

    def handle_workers_return(self, future):
        worker_id = future.result()
        print(f"Parallel: Worker {worker_id}'s response has been receive.")

        self.buff_lock.acquire()
        self.states_buff[self._buff_id] = self.workers_state_data_buff[worker_id].copy()
        self.path_buff[self._buff_id] = self.workers_path_buff[worker_id].copy()
        self._buff_id += 1

        if self._buff_id >= self.batch_size:
            self.update_states()
        self.buff_lock.release()

    def update_states(self):
        print(f"Parallel: Update Parallel.states")

        # print(self.states_buff)
        # model prediction -> Require engine or engine.history, engine.captures
        # backprop

        self._buff_id = 0
        # Return worker reference to start a new iteration with this thread

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
            Gomoku(),
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
