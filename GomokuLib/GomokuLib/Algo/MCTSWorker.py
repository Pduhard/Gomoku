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


path_dtype = np.dtype([
    ('statehash', np.dtype('<U361')),
    ('player_idx', np.int32),
    ('bestaction', np.int32, (2,)),
])
path_nb_dtype = nb.from_dtype(path_dtype)

path_array_dtype = nb.types.Array(dtype=path_dtype, ndim=361, layout='C')
print(path_array_dtype)

leaf_data_dtype = np.dtype([
    ('Visits', np.int32),
    ('Rewards', np.float32),
    ('StateAction', np.float32, (2, 19, 19)),
    ('Actions', np.int32, (2, 19, 19)),
])
leaf_data_nb_dtype = nb.from_dtype(leaf_data_dtype)

states_nb_dtype = nb.typeof(nb.typed.Dict.empty(
    key_type=nb.types.unicode_type,
    value_type=leaf_data_nb_dtype
))

worker_ret_dtype = np.dtype([
    ('worker_id', np.int32),
    ('leaf_data', leaf_data_dtype),
    ('path', path_array_dtype),
    ('depth', np.int32)
])
worker_ret_dtype = np.int64
worker_ret_nb_dtype = nb.from_dtype(worker_ret_dtype)


@jitclass()
class MCTSWorker:
    """
        Currently mix-in of MCTS() and MCTSLazy()
    """

    id: nb.types.int32
    engine: GomokuJit
    c: nb.types.float32

    # states: states_nb_dtype
    state_data_buff: worker_ret_nb_dtype

    # end_game: nb.types.boolean

    def __init__(self, 
                engine: GomokuJit,
                id: nb.types.int32,
                ):
        print(f"Worker id={self.id} __init__()\n")

        self.id = id
        self.engine = engine.clone()
        self.c = np.sqrt(2)

        # self.states = {'a': np.zeros((), dtype=leaf_data_dtype)}

        self.state_data_buff = np.zeros(1, dtype=worker_ret_dtype)
        # self.state_data_buff = np.arange(np.int64(1), dtype=worker_ret_dtype)
        # self.state_data_buff['worker_id'] = self.id

    def __str__(self):
        return f"MCTSWorker id={self.id}"

    def do_your_fck_work(self) -> tuple:
        print(f"Worker id={self.id} do_your_fck_work()")

        self.state_data_buff['leaf_data']['Visits'] = 111111
        self.state_data_buff['path'][0]['statehash'] = 'blablablablabla'
        self.state_data_buff['depth'] = 1

        # return self.state_data_buff
        return 1

    def mcts(self):

        # print(f"\n[MCTS function {mcts_iter}]\n")

        # self.path_player_idx = [0]
        self.best_action = (0, 0)

        statehash = self.engine.board.tobytes()
        self.end_game = self.engine.isover()
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

        # sleep(2)
        return 0

    def get_actions(self) -> nb.int32[:, :]:
        return self.engine.full_board ^ 1

    def get_policy(self, state_data) -> nb.int32[:, :]:
        """
            ucb(s, a) = exploitation_rate(s, a) + exploration_rate(s, a)

            exploitation_rate(s, a) =   reward(s, a) / (visits(s, a) + 1)
            exploration_rate(s, a) =    c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )

        """
        s_v = state_data['Visits']
        sa_v, sa_r = state_data['StateAction']
        sa_v = sa_v + 1
        return sa_r / sa_v + self.c * np.sqrt(np.log(s_v) / sa_v)

    def expand(self):
        actions = self.get_actions()
        self.reward = self.award_end_game() if self.end_game else self.award()
        return {
            'Visits': 1,
            'Rewards': 0,
            'StateAction': np.zeros((2, self.brow, self.bcol)),
            'Actions': actions,
        }

    def award(self):
        return 0.5

    def award_end_game(self):
        if self.draw:
            return 0.5
        return 1 if self.engine.winner == self.engine.player_idx else 0
