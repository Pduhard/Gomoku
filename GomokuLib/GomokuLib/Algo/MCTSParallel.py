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
                 num_workers: int = 1,
                 batch_size: int = 3,
                 pool_num: int = 2,
                 mcts_iter: int = 9,
                 *args, **kwargs
                 ) -> None:
        """
            3 workers | n pools de buffers
            [[_, _, _], ..., [_, _, _]]

            Workers acquire les buffers dans l'ordre.
            Dès qu'une pool est fini les workers pointent vers la suivante
                Main thread release tous les buffers de la pool
                et expand / backprop / ...
        """
        # super().__init__(engine, *args, **kwargs)

        self.engine = engine.clone()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mcts_iter = mcts_iter

        self.pool_num = pool_num
        self.buff_num = self.batch_size
        self.buff_id = 0
        self.pool_id = 0

        self.states = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        self.states_buff = np.recarray(
            shape=(self.pool_num, self.buff_num),
            dtype=Typing.StateDataDtype
        )
        self.path_buff = np.recarray(
            shape=(self.pool_num, self.buff_num, 361),
            dtype=Typing.PathDtype
        )
        self.pools_locks = [
            threading.Lock()
            for _ in range(self.pool_num)
        ]
        self.buffers_locks = [
            [
                threading.Lock()
                for _ in range(self.buff_num)
            ]
            for _ in range(self.pool_num)
        ]

        print(f"Parallel __init__() shapes: {self.states_buff.shape}\t {self.path_buff.shape}")
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

        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=None,
        )
        self.workers = [    # Autant de workers qu'il y a de Threads (-1 ?)
            MCTSWorker(
                np.int32(i),
                engine,
                self.states_buff,
                self.path_buff,
                self.states
            )
            for i in range(self.num_workers)
        ]
        print(f"Parallel: Successfully create {len(self.workers)} workers")

    def __str__(self):
        return f"MCTSParallel with {self.num_workers} workers and  iterations"

    def __call__(self) -> tuple:
        """
            Tester si ya des proiblem avec plusieurs thread sur la meme instance de classjit
            IL FAUT LES WORKER
        """
        print(f"\n[MCTSParallel begin __call__()] -> {self.num_workers} workers for {self.mcts_iter} iter\n")

        self.pool_id = 0
        self.buff_id = 0
        # Submit all Workers for an iteration
        not_done_futures = []
        for i, worker in enumerate(self.workers):
            self.buffers_locks[self.pool_id][i].acquire()
            not_done_futures.append(self.pool.submit(worker.do_your_fck_work, self.pool_id, i))

        # Main loop
        not_done_futures = not_done_futures
        _iter = len(not_done_futures) - 1
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
                # Search the first release buff of the pool self.pool_id
                # while self.buff_id < self.buff_num and not :
                #     print(f"Parallel: SEARCH | pool {self.pool_id} buff {self.buff_id} | locked ?-> {self.buffers_locks[self.pool_id][self.buff_id].locked()}")
                #     self.buff_id += 1

                if self.buffers_locks[self.pool_id][self.buff_id].locked():
                    print(f"===================================================== Buffer lock ??????????\n")
                self.buffers_locks[self.pool_id][self.buff_id].acquire()
                self.buff_id += 1

                if self.buff_id == self.buff_num:
                    f = self.pool.submit(self.update_states, self.pool_id)
                    not_done_futures.append(f)
                    self.next_pool()
                    self.buffers_locks[self.pool_id][self.buff_id].acquire(blocking=True)

                print(f"Parallel: SUBMIT WORKER {future.result()} | pool {self.pool_id} buff {self.buff_id}\n")
                # Submit done Workers for a new iteration
                f = self.pool.submit(self.workers[future.result()].do_your_fck_work, self.pool_id, self.buff_id)
                # f.add_done_callback(self.handle_workers_return)
                not_done_futures.append(f)
                _iter += 1

        # Wait until all workers has finished
        concurrent.futures.wait(
            not_done_futures,
            return_when=concurrent.futures.ALL_COMPLETED
        )
        # S'occuper de la dernière pool pas remplie ! ###################################################

        print(f"\n[MCTSParallel end __call__()] -> {self.num_workers} workers for {self.mcts_iter} iter\n")
        time.sleep(2)
        print(f"self.states length: {len(self.states.keys())}")

    def next_pool(self):
        pool_id = self.pool_id + 1
        if pool_id == self.pool_num:
            pool_id = 0
        if self.pools_locks[pool_id].locked():
            print(f"===================================================== POOL lock ??????????\n")
        self.pools_locks[pool_id].acquire()
        self.pool_id = pool_id      # After acquire()
        self.buff_id = 0

    def update_states(self, pool_id: int, buff_size: int):
        """
            ThreadPoolExecutor docs:
                Added callables are called in the order that they were added and
                are always called in a thread belonging to the process that added them.

            stackoverflow:
                If the thread is not cancelled:
                    Callback function execute by the thread that executes the future's task

            Vu que c'est ce fou de main thread qui prends en charge les data
            des workers_buff dans le callback, les workers peuvent overwrite leurs buffer
            alors qu'on ne les a pas encore "vidé" !
                -> Faire un lock par buffer[i]
                    -> Les workers ne seront plus ralenti par leur buffer non vidé !
                    Car ils prennent le premier buff qui est pas .acquire()
        """
        print(f"Parallel: Update Parallel.states with pool {pool_id}")

        for i in range(buff_size):
            path_len = self.states_buff[pool_id, i].depth
            k = str(self.path_buff[pool_id, i, path_len - 1].board.tobytes())

            print(f"Parallel: Update states with pool {pool_id} buff {i} + lock.release()")
            # Expansion
            if k in self.states:
                print(f"=============================== WTF EXPAND AN EXISTING STATE: {np.argwhere(self.path_buff[pool_id, i, path_len - 1].board == 1)}")
            self.states[k] = np.recarray(1, dtype=Typing.StateDataDtype)
            self.states[k][0] = self.states_buff[pool_id, i]

            # model prediction -> Require engine or engine.history, engine.captures
            # self.backpropagation(
            #     self.path_buff[pool_id, i],
            #     path_len,
            #     self.states_buff[pool_id, i].heuristic
            # )
            self.buffers_locks[pool_id][i].release()

        self.pools_locks[pool_id].release()
        return 0

    def backpropagation(self, path: np.ndarray, path_len: int, reward: Typing.MCTSFloatDtype):

        for i in range(path_len - 1, -1, -1):
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
