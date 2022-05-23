import concurrent.futures
import threading
import time

import numpy as np

import numba as nb
from numba import njit, prange
from numba.experimental import jitclass
import faulthandler

import GomokuLib.Typing as Typing

from GomokuLib.Algo.MCTS import MCTS
from GomokuLib.Algo.MCTSLazy import MCTSLazy
from GomokuLib.Algo.MCTSWorker import MCTSWorker
from GomokuLib.Algo.MCTSEvalLazy import MCTSEvalLazy
from GomokuLib.Game.GameEngine import Gomoku


class MCTSParallel:

    def __init__(self,
                 engine: Gomoku,
                 num_workers: int = 12,
                 batch_size: int = 32,
                 pool_num: int = 10,
                 mcts_iter: int = 5000,
                 *args, **kwargs
                 ) -> None:
        """
            n pools de n buffers
            [[_, _, _], ..., [_, _, _]]

            Dès qu'une pool est fini:
                On update les states avec celle-ci
                On la release
                On acquire la suivante pour que les workers la remplisse



            gdb -ex run --args python sandbox.py :
                0x00007fff52965392

                Thread 3 "python" received signal SIGSEGV, Segmentation fault.
                raise (sig=<optimized out>) at ../sysdeps/unix/sysv/linux/raise.c:50
                50      ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.

                Thread 3 "python" received signal SIGABRT, Aborted.
                __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
                50      ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.

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
        self.states_buff[...].worker_id = -1

        self.path_buff = np.recarray(
            shape=(self.pool_num, self.buff_num, 361),
            dtype=Typing.PathDtype
        )

        self.pools_locks = [
            threading.Lock()
            for _ in range(self.pool_num)
        ]
        self.check_pools_lock = threading.Lock()
        self.n_pool_acquire = 0
        self.n_pool_release = 0

        engine = Gomoku()

        self.threads_num = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers + 1,
            initializer=self.init_thread
        )
        # self.pool = concurrent.futures.ProcessPoolExecutor(
        #     max_workers=1,
        #     initializer=self.init_thread
        # )
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
        print(f"Parallel: __init__() end. Shapes: {self.states_buff.shape}\t {self.path_buff.shape}")

    def __str__(self):
        return f"MCTSParallel with {self.num_workers} workers and  iterations"

    def __call__(self, game_engine: Gomoku) -> tuple:
        self.game_engine = game_engine

        faulthandler.enable(all_threads=True)   # Print traceback of segfaults
        self.work()

        k = self.tobytes(self.game_engine.board)
        state_data = self.states[k][0]
        sa_v, sa_r = state_data.stateAction
        sa_v += 1
        arg = np.argmax(sa_r / sa_v)
        # print(f"StateAction visits: {sa_v}")
        # print(f"StateAction reward: {sa_r}")
        # print(f"Qualities: {sa_r / sa_v}")
        print(f"argmax: {arg} / {int(arg / 19)} {arg % 19}")
        return int(arg / 19), arg % 19

    def work(self):
        """
            Tester si ya des proiblem avec plusieurs thread sur la meme instance de classjit
            IL FAUT LES WORKER
        """
        print(f"\n[MCTSParallel begin __call__()] -> {self.num_workers} workers for {self.mcts_iter} iter\n")

        print(f"self.pools_locks locked: {[p.locked() for p in self.pools_locks]}")
        for p in self.pools_locks:
            if p.locked():
                p.release()

        assert not self.check_pools_lock.locked()
        assert not any(l.locked() for l in self.pools_locks) # All pool locks release
        assert not np.any(self.states_buff.worker_id != -1) # All worker_id in state_buff at -1

        _iter = 0
        self.pool_id = 0
        self.buff_id = 0

        self.pools_locks[self.pool_id].acquire()
        self.n_pool_acquire = 1
        self.n_pool_release = 0

        # Submit all Workers for an iteration
        not_done_futures = set()   # Set() object
        for worker_id in range(len(self.workers)):
            f = self.submit_worker(worker_id)
            not_done_futures.add(f)
            _iter += 1

        # Submit done Workers for next iterations
        for future in concurrent.futures.as_completed(not_done_futures):
            f = self.submit_worker(future.result())
            not_done_futures.add(f)
            _iter += 1
            # print(f"mcts iter: {_iter}")

        # Wait until all workers has finished
        concurrent.futures.wait(
            not_done_futures,
            return_when=concurrent.futures.ALL_COMPLETED
        )

        # Process all buff, no longer 'NOT FINISHED', of last pool
        self.check_pool_state(None, force_update=True)

        time.sleep(1)
        self.pools_locks[self.pool_id].release()
        print(f"\n[MCTSParallel end __call__()] -> {self.num_workers} workers for {self.mcts_iter} iter\n")
        print(f"self.states length: {len(self.states.keys())}")
        print(f"Max thread used: {self.threads_num}")
        print(f"Pool acquire / release: {self.n_pool_acquire} / {self.n_pool_release}")

    def init_thread(self):
        # print(f"Init thread by {threading.current_thread().name}")
        self.threads_num += 1

    def get_state_data(self, engine):
        return {
            'mcts_state_data': self.states[self.tobytes(self.engine.board)],
        }

    def get_state_data_after_action(self, engine):
        return {}

    def reset(self):
        self.states = {}

    def submit_worker(self, worker_id: int) -> concurrent.futures:

        # Submit a worker
        # print(f"Parallel: SUBMIT WORKER {worker_id} | pool {self.pool_id} buff {self.buff_id}")
        worker_f = self.pool.submit(self.workers[worker_id].do_your_fck_work, self.game_engine, self.pool_id, self.buff_id)

        # Callback: If all buff of a pool are no longer 'NOT FINISHED', process these buffers
        worker_f.add_done_callback(self.check_pool_state)

        # If the pool is full, go to the next pool
        self.buff_id += 1
        if self.buff_id == self.buff_num:
            self.next_pool()

        return worker_f

    def next_pool(self):
        # Compute next pool_id
        pool_id = self.pool_id + 1
        if pool_id == self.pool_num:
            pool_id = 0

        # Acquire related lock instant, or wait for it
        # if self.pools_locks[pool_id].locked():
        #     print(f"===================================================== POOL lock ?????????? Need more pools !\n")
        try:
            self.pools_locks[pool_id].acquire()
            pass
        except:
            print(f"===================================================== pool({pool_id}).acquire() SEG FAULT !!?\n")
            breakpoint()

        self.n_pool_acquire += 1

        # Update indexes
        self.pool_id = pool_id      # After acquire()
        self.buff_id = 0
        # print(f"======================= Lock pool {pool_id} !")

    def check_pool_state(self, future: concurrent.futures.Future, force_update: bool = False):
        """
            If all buff of a pool are no longer 'NOT FINISHED', process these buffers

            ThreadPoolExecutor docs about callbacks:
                Added callables are called in the order that they were added and
                are always called in a thread belonging to the process that added them.

            Random guy on stackoverflow, about callbacks:
                If the thread is not cancelled:
                    Callback function execute by the thread that executes the future's task
        """
        # worker_id = future.result() if future else None

        # path_len = self.states_buff[0, worker_id].depth
        # print(f"Worker {worker_id}, path depth {path_len} returns:\n{self.path_buff[0, worker_id, path_len - 1]}")
        # print(f"Parallel: CALLBACK WORKER {worker_id}")

        self.check_pools_lock.acquire()
        for i, pool in enumerate(self.states_buff):

            finished = pool[:].worker_id != -1
            if (all(finished) or
               (force_update and any(finished))):
                # print(f"pool {i}, finished workers:\n{finished}")
                # Reset all buff to an imaginary state 'NOT FINISHED'
                self.states_buff[i, :].worker_id = -1
                # Update self.states with buffers
                self.pool.submit(self.update_states, i, np.count_nonzero(finished))
        self.check_pools_lock.release()

    def update_states(self, pool_id: int, buff_size: int):
        # print(f"\nParallel: Update Parallel.states with {buff_size} buff of pool {pool_id} by thread {threading.current_thread().name}")

        for i in range(buff_size):
            path_len = self.states_buff[pool_id, i].depth
            # k = str(self.path_buff[pool_id, i, path_len - 1].board.tobytes())
            k = self.tobytes(self.path_buff[pool_id, i, path_len].board)

            # Expansion
            # if k in self.states:
            #     # print(f"=============================== WTF EXPAND AN EXISTING STATE: {np.argwhere(self.path_buff[pool_id, i, path_len - 1].board == 1)}")
            #     # print(self.path_buff[pool_id, i, path_len - 1].board)
            #     # print(k)
            #     breakpoint()
            self.states[k] = np.recarray(1, dtype=Typing.StateDataDtype)
            self.states[k][0] = self.states_buff[pool_id, i]

            # model prediction -> Require engine or engine.history, engine.captures
            self.backpropagation(
                self.path_buff[pool_id, i],
                path_len,
                self.states_buff[pool_id, i].heuristic
            )

        # Release the pool
        if self.pools_locks[pool_id].locked():
            self.pools_locks[pool_id].release()
            self.n_pool_release += 1
            # print(f"========================== Release pool {pool_id} !\n")
        else:
            # print(f"============================ Wtf lock pool {pool_id} was already release !")
            breakpoint()

    def backpropagation(self, path: np.ndarray, path_len: int, reward: Typing.MCTSFloatDtype):

        for i in range(path_len - 1, -1, -1):
            # print(f"Parallel: backprop index of path: {i}")
            self.backprop_memory(path[i], reward)
            reward = 1 - reward

    def backprop_memory(self, memory: np.ndarray, reward: Typing.MCTSFloatDtype):
        # print(f"Memory:\n{memory}")
        # print(f"Memory dtype:\n{memory.dtype}")
        board = memory.board
        bestAction = memory.bestAction

        # state_data = self.states[str(board.tobytes())]
        state_data = self.states[self.tobytes(board)][0]

        state_data.visits += 1                           # update n count
        state_data.rewards += reward                     # update state value
        if bestAction[0] == -1:
            return

        r, c = bestAction
        state_data.stateAction[..., r, c] += [1, reward]  # update state-action count / value

    def tobytes(self, arr: Typing.nbBoard):
        return ''.join(map(str, map(np.int8, np.nditer(arr))))
