from time import sleep
import numpy as np
import numba as nb

from numba.experimental import jitclass

import GomokuLib.Typing as Typing
from GomokuLib.Game.GameEngine import Gomoku


@jitclass()
class MCTSWorker:
    """
        Currently mix-in of MCTS() and MCTSLazy()

        Passer un state_data buffer en recarray du parralel aux workers avec l'id ou ils
        doivent l'ecrire. -> state_data_buff[buff_id][:] = state_data
            Idem avec le path ->   path_buff[buff_id][:] = path

        Ensuite Dans le parallel :
            Save le return du worker dans un autre buffer de taille batch_size
            Quand ce buff est plein, update le state global

    """

    id: nb.int32
    engine: Gomoku
    c: nb.float32

    # states: Typing.nbStates[:]

    state_data_buff: Typing.nbStates
    path_buff: Typing.nbPath
    empty_state_data: Typing.nbStateArray

    # end_game: nb.types.boolean

    def __init__(self, 
                 id: nb.int32,
                 engine: Gomoku,
                 state_data_buff: Typing.nbStates,
                 path_buff: Typing.nbPath
                 ):

        self.id = id
        self.engine = engine.clone()
        self.state_data_buff = state_data_buff
        self.path_buff = path_buff

        self.c = np.sqrt(2)

        # self.empty_state_data = np.array(
        #     [(self.id, 0, 0, 0, np.zeros((2, 19, 19)), np.zeros((19, 19), 0.5))],
        #     dtype=Typing.StateDataDtype
        # )
        empty_state_data = np.zeros(
            shape=1,
            dtype=Typing.StateDataDtype,
        )[0]
        # self.empty_state_data = np.recarray((1,), dtype=Typing.StateDataDtype, buf=self.empty_state_data)
        empty_state_data.Worker_id = self.id
        empty_state_data.Visits = 1

        print(empty_state_data)
        print(f"Worker {self.id}: end __init__()\n")

    def __str__(self):
        return f"MCTSWorker id={self.id}"

    def do_your_fck_work(self) -> tuple:
        print(f"Worker {self.id}: do_your_fck_work()")

        self.state_data_buff[self.id] = np.zeros(
            shape=1,
            dtype=Typing.StateDataDtype,
        )[0]
        path = np.ones_like(self.path_buff[self.id].board)
        # self.state_data_buff[self.id].

        self.path_buff[self.id].board[...] = path

        # return self.state_data_buff
        return self.id

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
    #
    # def get_actions(self) -> nb.int32[:, :]:
    #     return self.engine.full_board ^ 1
    #
    # def get_policy(self, state_data) -> nb.int32[:, :]:
    #     """
    #         ucb(s, a) = exploitation_rate(s, a) + exploration_rate(s, a)
    #
    #         exploitation_rate(s, a) =   reward(s, a) / (visits(s, a) + 1)
    #         exploration_rate(s, a) =    c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )
    #
    #     """
    #     s_v = state_data['Visits']
    #     sa_v, sa_r = state_data['StateAction']
    #     sa_v = sa_v + 1
    #     return sa_r / sa_v + self.c * np.sqrt(np.log(s_v) / sa_v)
    #
    # def expand(self):
    #     actions = self.get_actions()
    #     self.reward = self.award_end_game() if self.end_game else self.award()
    #     return {
    #         'Visits': 1,
    #         'Rewards': 0,
    #         'StateAction': np.zeros((2, self.brow, self.bcol)),
    #         'Actions': actions,
    #     }
    #
    # def award(self):
    #     return 0.5
    #
    # def award_end_game(self):
    #     if self.draw:
    #         return 0.5
    #     return 1 if self.engine.winner == self.engine.player_idx else 0
