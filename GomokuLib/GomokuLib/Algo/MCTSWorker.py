import time
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
        doivent l'ecrire. -> states_buff[buff_id][:] = state_data
            Idem avec le path ->   path_buff[buff_id][:] = path

        Ensuite Dans le parallel :
            Save le return du worker dans un autre buffer de taille batch_size
            Quand ce buff est plein, update le state global

    """

    id: nb.int32
    engine: Gomoku
    c: nb.float32

    states: Typing.nbStateDict

    states_buff: Typing.nbStateBuff
    path_buff: Typing.nbPath
    # empty_state_data: Typing.nbStateArray

    # end_game: nb.types.boolean

    def __init__(self, 
                 id: nb.int32,
                 engine: Gomoku,
                 states_buff: Typing.nbStateBuff,
                 path_buff: Typing.nbPath,
                 states: Typing.nbStateDict,
                 ):

        self.id = id
        self.engine = engine.clone()
        self.states_buff = states_buff
        self.path_buff = path_buff
        self.states = states
        self.c = np.sqrt(2)

        print(f"Worker {self.id}: end __init__()\n")

    def __str__(self):
        return f"MCTSWorker id={self.id}"

    def do_your_fck_work(self, pool_id: int, buff_id: int) -> tuple:
        """
            ## Ca ca marche :
                path[0].bestaction[:] = ba
            ## Ca non !
                path[0].bestaction = ba

            ## It works ! At least from there ...
            state_data = np.zeros(1, dtype=Typing.StateDataDtype)
            state_data[0].worker_id = self.id
            state_data[0].depth = 6
            state_data[0].stateAction[...] = np.ones((2, 19, 19), dtype=Typing.MCTSFloatDtype)
            state_data[0].heuristic = 0.420

            path = np.zeros(1, dtype=Typing.PathDtype)
            path[0].board[...] = np.ones((2, 19, 19), dtype=Typing.BoardDtype)
            path[0].player_idx = Typing.MCTSIntDtype(42)
            path[0].bestAction[...] = np.ones(2, dtype=Typing.ActionDtype)

            self.states_buff[self.id] = state_data[0]
            self.path_buff[self.id] = path[0]
            ## ... to there :)
        """

        with nb.objmode():
            print(f"Worker {self.id}: do_your_fck_work() | pool {pool_id} buff {buff_id} | self.states length: {len(self.states.keys())}", flush=True)
            time.sleep(0.2)

        state_data = np.zeros(1, dtype=Typing.StateDataDtype)
        path = np.zeros(1, dtype=Typing.PathDtype)

        state_data[0].depth = 1

        rd = np.zeros(722, dtype=Typing.BoardDtype)
        ri = np.random.randint(722)
        print(ri)
        rd[ri] = 1
        path[0].board[...] = rd.reshape((2, 19, 19))

        ## Futur?: Envoyer state_data directement pour ne pas re-déclarer
        ## un array lors de l'insersion dans states
        self.states_buff[pool_id, buff_id] = state_data[0]
        self.path_buff[pool_id, buff_id] = path[0]
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
