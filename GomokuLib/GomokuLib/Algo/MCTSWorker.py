import time
from time import sleep

import GomokuLib
import numpy as np
import numba as nb

from numba.experimental import jitclass

import GomokuLib.Typing as Typing
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib.Algo.MCTSUtils import MCTSUtils


"""
@overload(np.any)
@overload_method(types.Array, "any")
def np_any(a):
    def flat_any(a):
        for v in np.nditer(a):
            if v.item():
                return True
        return False

    return flat_any
"""


# @jitclass()
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
    # MCTSUtils: MCTSUtils
    c: nb.float32
    pruning: nb.boolean

    states: Typing.nbStateDict

    states_buff: Typing.nbStateBuff
    path_buff: Typing.nbPathBuff

    leaf_data: Typing.nbState
    path: Typing.nbPath

    end_game: nb.boolean

    def __init__(self, 
                 id: nb.int32,
                 engine: Gomoku,
                 states_buff: Typing.nbStateBuff,
                 path_buff: Typing.nbPathBuff,
                 states: Typing.nbStateDict,
                 pruning: nb.boolean = True
                 ):

        self.id = id
        self.engine = engine.clone()
        self.states_buff = states_buff
        self.path_buff = path_buff
        self.states = states
        self.pruning = pruning

        self.leaf_data = np.zeros(1, dtype=Typing.StateDataDtype)
        self.path = np.zeros(361, dtype=Typing.PathDtype)

        self.c = np.sqrt(2)
        # self.MCTSUtils = MCTSUtils()
        # if self.pruning:
        #     self.get_policy = self.MCTSUtils.get_policy
        #     self.expand_variant = self.MCTSUtils.expand
        # else:
        #     self.expand_variant = self._expand_variant

        print(f"Worker {self.id}: end __init__()\n")

    def __str__(self):
        return f"MCTSWorker id={self.id}"

    def do_your_fck_work(self, game_engine: Gomoku, pool_id: Typing.MCTSIntDtype, buff_id: Typing.MCTSIntDtype) -> tuple:
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

            ## Futur?: Envoyer state_data directement pour ne pas re-déclarer
            ## un array lors de l'insersion dans states
        """
        # return self.MCTSParallel_tests(pool_id, buff_id)

        # print(f"\n[MCTS function pool {pool_id} buff {buff_id}]\n")

        self.engine.update(game_engine)
        self.mcts()

        self.path_buff[pool_id, buff_id, ...] = self.path
        self.states_buff[pool_id, buff_id] = self.leaf_data[0]

        # A faire le plus à la fin possible car MCTSParallel utilise cette valeur pour determiner la fin du MCTSWorker!
        self.states_buff[pool_id, buff_id].worker_id = self.id
        # if self.leaf_data[0].depth > 0:
        #     with nb.objmode():
        #         breakpoint()

        return self.id

    def mcts(self):

        depth = 0
        self.end_game = self.engine.isover()
        # statehash = str(self.engine.board.tobytes())
        statehash = self.tobytes(self.engine.board)
        while statehash in self.states and not self.end_game:

            state_data = self.states[statehash][0]

            policy = self.get_policy(state_data, self.c)
            best_action = self.selection(policy, state_data)

            self.fill_path(depth, best_action)
            self.engine.apply_action(best_action)
            self.engine.next_turn()
            depth += 1

            self.end_game = self.engine.isover()
            # statehash = str(self.engine.board.tobytes())
            statehash = self.tobytes(self.engine.board)

        self.fill_path(depth, np.full(2, -1, Typing.MCTSIntDtype))
        self.expand(depth)

    def get_policy(self, state_data: Typing.nbState, *args) -> Typing.nbPolicy:
        """
            ucb(s, a) = exploitation_rate(s, a) + exploration_rate(s, a)

            exploitation_rate(s, a) =   reward(s, a) / (visits(s, a) + 1)
            exploration_rate(s, a) =    c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )

        """
        s_v = state_data.visits
        sa_v, sa_r = state_data.stateAction
        sa_v += 1   # Init this value at 1 ?
        ucbs = sa_r / sa_v + self.c * np.sqrt(np.log(s_v) / sa_v)
        # if self.pruning:
        #     return ucbs * state_data.pruning
        return ucbs

    def selection(self, policy: np.ndarray, state_data: Typing.nbState) -> Typing.nbTuple:
        best_action = np.zeros(2, dtype=Typing.MCTSIntDtype)
        available_actions = policy * state_data.actions     # Avaible actions
        best_actions = np.argwhere(available_actions == np.amax(available_actions))
        action = best_actions[np.random.randint(len(best_actions))]
        best_action[0] = action[0]
        best_action[1] = action[1]
        # return Typing.nbTuple(best_action)
        return best_action

    def fill_path(self, depth: Typing.MCTSIntDtype, best_action: np.ndarray):
        self.path[depth].board[...] = self.engine.board
        self.path[depth].player_idx = self.engine.player_idx
        self.path[depth].bestAction[:] = best_action
        
    def expand(self, depth: Typing.MCTSIntDtype):
        self.leaf_data[0].depth = depth
        self.leaf_data[0].visits = 1
        self.leaf_data[0].actions[:] = self.engine.get_actions()
        self.leaf_data[0].heuristic = self.award_end_game() if self.end_game else self.award()

        # if self.pruning:
        #     self.leaf_data[0].pruning = MCTSUtils.pruning(self.engine)

        # Laggy to pass all self's data to MCTSUtils, better use expand_variant instead
        # self.expand_variant(self.leaf_data, self.engine)    # Like MCTSEval (Pruning)

    # def _expand_variant(self):
    #     pass

    def award_end_game(self):
        if self.engine.winner == -1: # Draw
            return 0.5
        return 1 if self.engine.winner == self.engine.player_idx else 0

    def tobytes(self, arr: Typing.nbBoard):
        return ''.join(map(str, map(np.int8, np.nditer(arr))))

    def award(self):
        return 0.5

    # def award(self):
    #     """
    #         Mean of leaf state heuristic & random(pruning) rollingout end state heuristic
    #     """
    #     h_leaf = self.MCTSUtils.heuristic(self.engine)
    #     if self.rollingout_turns:
    #         self._random_rollingout(self.rollingout_turns)
    #
    #         if self.engine.isover():
    #             return self.award_end_game()
    #         else:
    #             h = self.MCTSUtils.heuristic(self.engine)
    #             return (h_leaf + (1 - h if self.rollingout_turns % 2 else h)) / 2
    #     else:
    #         return h_leaf
    #
    # def rollingout(self, n_turns):
    #     gAction = np.zeros(2, dtype=Typing.TupleDtype)
    #     turn = 0
    #     while not self.engine.isover() and turn < n_turns:
    #
    #         pruning = self.MCTSUtils.pruning(self.engine).flatten().astype(np.bool8)
    #
    #         # Create actions from pruning
    #         if pruning.any():
    #             actions = self.all_actions[pruning > 0]
    #         else:
    #             actions = self.all_actions
    #
    #         # Select randomly an action from actions/pruning
    #         action_number = len(actions)
    #         i = np.random.randint(action_number)
    #         gAction = actions[i]
    #         while not self.engine.is_valid_action(gAction):
    #             i = np.random.randint(action_number)
    #             gAction = actions[i]
    #
    #         self.engine.apply_action(gAction)
    #         self.engine.next_turn()
    #         turn += 1

if __name__ == '__main__':

    pool_num = 1
    buff_num = 1
    path_buff = np.recarray(
        shape=(pool_num, buff_num, 361),
        dtype=Typing.PathDtype
    )
    states_buff = np.recarray(
        shape=(pool_num, buff_num),
        dtype=Typing.StateDataDtype
    )
    states_buff[...].worker_id = -1
    states = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=Typing.nbState
    )
    pool_id = 0
    i = 0

    def tobytes(arr: Typing.nbBoard):
        return ''.join(map(str, map(np.int8, np.nditer(arr))))

    def expand():
        path_len = states_buff[pool_id, i].depth
        k = tobytes(path_buff[pool_id, i, path_len].board)

        states[k] = np.recarray(1, dtype=Typing.StateDataDtype)
        states[k][0] = states_buff[pool_id, i]

        backpropagation(
            path_buff[pool_id, i],
            path_len,
            states_buff[pool_id, i].heuristic
        )

    def backpropagation(path: np.ndarray, path_len: int, reward):

        for i in range(path_len - 1, -1, -1):
            # print(f"Parallel: backprop index of path: {i}")
            backprop_memory(path[i], reward)
            reward = 1 - reward

    def backprop_memory(memory: np.ndarray, reward):
        # print(f"Memory:\n{memory}")
        # print(f"Memory dtype:\n{memory.dtype}")
        board = memory.board
        bestAction = memory.bestAction

        # state_data = self.states[str(board.tobytes())]
        state_data = states[tobytes(board)][0]

        state_data.visits += 1                           # update n count
        state_data.rewards += reward                     # update state value
        if bestAction[0] == -1:
            return

        r, c = bestAction
        state_data.stateAction[..., r, c] += [1, reward]  # update state-action count / value


    runner = GomokuLib.Game.GameEngine.GomokuRunner()

    mcts = GomokuLib.Algo.MCTSWorker(
        6,
        runner.engine,
        states_buff,
        path_buff,
        states
    )
    for k in range(500):
        ret = mcts.do_your_fck_work(runner.engine, 0, 0)
        expand()
        print(len(states))
