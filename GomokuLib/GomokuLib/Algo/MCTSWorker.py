from time import sleep

import typing
import numpy as np

import numba as nb
from numba import deferred_type
from numba.experimental import jitclass

# from ..Game.GameEngine import Gomoku

@jitclass()
class GomokuJit:
    board: nb.int32[:, :, :]
    full_board: nb.int32[:, :]
    
    def __init__(self) -> None:
        self.board = np.zeros((2, 19, 19), dtype=np.int32)
        self.full_board = np.zeros((19, 19), dtype=np.int32)

    def clone(self):
        return GomokuJit()

    def update(self, engine):
        pass

    # def update(self):
    #     pass

# Gomoku_type = deferred_type()
# Gomoku_type.define(GomokuJit.class_type.instance_type) # Ne marchera que quand Gomoku sera jit

state_data_dtype = np.dtype([
    ('Visits', np.int32),
    ('Rewards', np.float32),
    ('StateAction', np.float32, (2, 19, 19)),
    ('Actions', np.int32, (2, 19, 19)),
])
state_data_nb_type = nb.typeof(np.zeros(1, dtype=state_data_dtype))

states_dtype = nb.typeof(nb.typed.Dict.empty(
    key_type=nb.types.unicode_type,
    # value_type=nb.types.float32
    value_type=state_data_nb_type
))


# spec = {
#     'id': nb.types.int32,
#     'engine': GomokuJit.class_type.instance_type,
#     'states': states_dtype,
# }

@jitclass()
class MCTSWorker:
    """
        Currently mix-in of MCTS() and MCTSLazy()
    """

    id: nb.types.int32
    engine: GomokuJit
    states: states_dtype
    c: nb.types.float32
    
    best_action: nb.typeof(nb.types.Tuple)
    # path_player_idx: nb.types.Array(np.int32, 361, 'C')
    # path_: nb.types.Array(np.int32, 361, 'C')
    # path_player_idx: nb.types.Array(np.int32, 361, 'C')


    def __init__(self, 
                engine: GomokuJit,
                id: nb.types.int32 = None):
        print(f"Worker id={self.id} __init__()\n")

        self.id = id
        self.engine = engine.clone()
        self.states = {'a': 0}
        # self.states = {self.engine.state.board: None}
        self.c = np.sqrt(2)

    def __str__(self):
        return f"MCTSWorker id={self.id} ( iter)"

    def do_your_fck_work(self,
                game_engine: GomokuJit,
                global_states: states_dtype,
                mcts_iter: nb.types.int32) -> tuple:
        print(f"Worker id={self.id} do_your_fck_work()")

        self.states = global_states
        self.engine.update(game_engine)

        worker_data = self.mcts(mcts_iter)

        return self, self.id, worker_data

    def mcts(self, mcts_iter: int):

        # print(f"\n[MCTS function {mcts_iter}]\n")

        # self.path_player_idx = [0]
        self.best_action = (0, 0)

        # statehash = self.engine.state.board.tobytes()
        # self.end_game = self.engine.isover()
        # while statehash in self.states and not self.end_game:

        #     state_data = self.states[statehash]

        #     policy = self.get_policy(state_data)
        #     self.best_action = self.selection(policy, state_data)

        #     path.append(self.new_memory(statehash))
        #     self.engine.apply_action(self.best_action)
        #     self.engine.next_turn()

        #     statehash = self.engine.state.board.tobytes()

        #     self.end_game = self.engine.isover()

        # self.draw = self.engine.winner == -1

        # self.best_action = None
        # path.append(self.new_memory(statehash))

        # return self.expand()

        # sleep(2)
        return 0

    def get_actions(self) -> nb.int32[:, :]:
        return self.engine.full_board ^ 1

    def get_policy(self, state_data: state_data_nb_type) -> nb.int32[:, :]:
        """
            ucb(s, a) = exploitation_rate(s, a) + exploration_rate(s, a)

            exploitation_rate(s, a) =   reward(s, a) / (visits(s, a) + 1)
            exploration_rate(s, a) =    c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )

        """
        s_v = state_data['Visits']
        sa_v, sa_r = state_data['StateAction']
        sa_v = sa_v + 1
        return sa_r / sa_v + self.c * np.sqrt(np.log(s_v) / sa_v)

    # def selection(self, policy: np.ndarray, state_data: list) -> tuple:

    #     actions = state_data['Actions']
    #     while True:
    #         # best_action_count = fastcore.mcts_lazy_selection(c_policy, self.c_best_actions_buffer)
    #         arr = test_selection_parallel(actions, policy)
    #         len = arr[-1, 0]
    #         arr_pick = np.arange(len)
    #         np.random.shuffle(arr_pick)

    #         for e in arr_pick:
    #             x, y = arr[e]
    #             gAction = (x, y)

    #             if actions[x, y] == 2:
    #                 return gAction
    #             elif self.engine.is_valid_action(gAction):
    #                 actions[x, y] = 2
    #                 return gAction
    #             else:
    #                 actions[x, y] = 0

    #     raise Exception("No valid action to select.")
