import concurrent.futures
import threading
import time

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
                 batch_size: int = 1,
                 mcts_iter: int = 11,
                 *args, **kwargs
                 ) -> None:
        # super().__init__(engine, *args, **kwargs)

        self.engine = engine.clone()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mcts_iter = mcts_iter

        self.buff_lock = threading.Lock()
        self.buff_size = self.batch_size
        self.buff_id = 0

        self.states = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        self.states_buff = np.recarray(
            shape=(self.buff_size, ),
            dtype=Typing.StateDataDtype
        )
        self.workers_state_data_buff = np.recarray(
            shape=(self.num_workers, ),
            dtype=Typing.StateDataDtype
        )
        self.path_buff = np.recarray(
            shape=(self.buff_size, 361),
            dtype=Typing.PathDtype
        )
        self.workers_path_buff = np.recarray(
            shape=(self.num_workers, 361),
            dtype=Typing.PathDtype
        )

        print(f"Parallel __init__() shapes: {self.states_buff.shape}\t {self.path_buff.shape}")
        print(f"Parallel __init__() shapes: {self.workers_state_data_buff.shape}\t {self.workers_path_buff.shape}")

        engine = Gomoku()

        ## ca marche
        ## Valeurs random quand on passe de nbState à state_data_nb_dtype
        ## en enlevant le [0] ...
        # bd = np.zeros((2, 19, 19), dtype=np.uint8)
        # k = str(bd.tobytes())
        # self.states[k] = np.recarray(1, dtype=Typing.StateDataDtype)
        #
        # a = np.zeros((19, 19), dtype=Typing.MCTSIntDtype)
        # self.states[k][0].actions = a
        # self.states[k][0].actions[0, 0] = 42
        # print(f"Temoin {self.states[k][0].actions[0, 0]}")
        ## ca marche pas encore ^^"

        ############################

        # bd = np.zeros((2, 19, 19), dtype=np.uint8)
        # self.states[bd.tobytes()] = np.recarray(1, dtype=Typing.StateDataDtype)
        # self.states[bd.tobytes()][0].Actions[-1, -1] = 42

        ############################

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
        concurrent.futures.wait(
            not_done_futures,
            return_when=concurrent.futures.ALL_COMPLETED
        )

        print(f"\n[MCTSParallel end __call__()] -> {self.num_workers} workers for {self.mcts_iter} iter\n")

    def handle_workers_return(self, future):
        """
            ThreadPoolExecutor docs:
                Added callables are called in the order that they were added and
                are always called in a thread belonging to the process that added them.

            stackoverflow:
                If the thread is not cancelled:
                    Callback function execute by the thread that executes the future's task
        """
        worker_id = future.result()
        print(f"Parallel: Worker {worker_id}'s response has been receive: {time.time()}")

        # print(f"self.workers_state_data_buff[worker_id]:\n{self.workers_state_data_buff[worker_id]}")
        # print(f"self.workers_path_buff[worker_id]:\n{self.workers_path_buff[worker_id]}")

        ## Futur?: Enlever les copy
        buff_size = self.workers_state_data_buff[worker_id].depth

        self.buff_lock.acquire()
        print(f"Worker {worker_id} acquire: {time.time()}")
        self.states_buff[self.buff_id] = self.workers_state_data_buff[worker_id].copy()
        self.path_buff[self.buff_id, :buff_size] = self.workers_path_buff[worker_id, :buff_size].copy()

        self.buff_id += 1
        if self.buff_id >= self.buff_size:
            self.update_states()
        self.buff_lock.release()
        print(f"Worker {worker_id} release: {time.time()}\n")

    def update_states(self):
        print(f"Parallel: Update Parallel.states: {self.buff_id} >= {self.buff_size} = {self.buff_id >= self.buff_size}")

        # Expansion
        for i in range(self.buff_size):
            k = str(self.path_buff[i, -1].board.tobytes())
            self.states[k] = np.recarray(1, dtype=Typing.StateDataDtype)
            self.states[k][0] = self.states_buff[i]

            self.backpropagation(
                self.path_buff[i],
                self.states_buff[i].depth,
                self.states_buff[i].heuristic
            )
        # model prediction -> Require engine or engine.history, engine.captures
        # backprop

        self.buff_id = 0
        print(f"Parallel: Set self.buff_id to 0 -> {self.buff_id}")
        # Return worker reference to start a new iteration with this thread

    def backpropagation(self, path: np.ndarray, path_size: int, reward: Typing.MCTSFloatDtype):

        for i in range(path_size - 1, -1, -1):
            # print(f"Parallel: backprop index of path: {i}")
            # self.backprop_memory(path[i], reward)
            reward = 1 - reward

    def backprop_memory(self, memory: np.ndarray, reward: Typing.MCTSFloatDtype):
        # print(f"Memory:\n{memory}")
        # print(f"Memory dtype:\n{memory.dtype}")
        board = memory.board
        bestAction = memory.bestAction

        state_data = self.states[str(board.tobytes())]

        state_data.visits += 1                           # update n count
        state_data.rewards += reward                     # update state value
        if bestAction is None:
            return

        r, c = bestAction.action
        state_data.stateAction[..., r, c] += [1, reward]  # update state-action count / value


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
