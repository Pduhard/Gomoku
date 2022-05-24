import time
from time import sleep

import GomokuLib
import numpy as np
import numba as nb

from numba import njit

from numba.experimental import jitclass

import GomokuLib.Typing as Typing
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib.Algo.MCTSUtils import MCTSUtils

import fastcore
from fastcore._algo import ffi, lib as fastcore

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


# @nb.vectorize('float64(int8, float64)')
# def valid_action(actions, policy):
#     if actions > 0:
#         return policy
#     else:
#         return 0

@njit()
def heuristic(engine):
    board = engine.board.astype(Typing.BoardDtype)
    full_board = (board[0] | board[1]).astype(Typing.BoardDtype)
    c_board = ffi.from_buffer(board)
    c_full_board = ffi.from_buffer(full_board)
    # c_full_board = ffi.cast("char *", full_board.ctypes.data)
    # if not engine.board.flags['C_CONTIGUOUS']:
    #     print(f"NOT continuoueo_iyfhg_uièyergbiuybziruygbirzuy")
    #     engine.board = np.ascontiguousarray(engine.board)
    # if not engine.full_board.flags['C_CONTIGUOUS']:
    #     print(f"NOT continuoueo_iyfhg_uièyergbiuybziruygbirzuy 2")
    #     engine.full_board = np.ascontiguousarray(engine.full_board)

    x = fastcore.mcts_eval_heuristic(
        c_board, c_full_board,
        *engine.get_captures(),
        *engine.get_game_zone()
    )
    return x


@njit()
def get_neighbors_mask(board):

    neigh = np.zeros((19, 19), dtype=board.dtype)

    neigh[:-1, :] |= board[1:, :]   # Roll cols to left
    neigh[1:, :] |= board[:-1, :]   # Roll cols to right
    neigh[:, :-1] |= board[:, 1:]   # Roll rows to top
    neigh[:, 1:] |= board[:, :-1]   # Roll rows to bottom

    neigh[1:, 1:] |= board[:-1, :-1]   # Roll cells to the right-bottom corner
    neigh[1:, :-1] |= board[:-1, 1:]   # Roll cells to the right-upper corner
    neigh[:-1, 1:] |= board[1:, :-1]   # Roll cells to the left-bottom corner
    neigh[:-1, :-1] |= board[1:, 1:]   # Roll cells to the left-upper corner

    return neigh

@njit()
def njit_prunning(engine, hard_pruning = True):

    full_board = (engine.board[0] | engine.board[1]).astype(np.bool8)
    n1 = get_neighbors_mask(full_board)                      # Get neightbors, depth=1

    if hard_pruning:
        non_pruned = n1
    else:
        n2 = get_neighbors_mask(n1)                         # Get neightbors, depth=2
        non_pruned = np.logical_or(n1, n2)
    xp = non_pruned ^ full_board
    non_pruned = xp & non_pruned  # Remove neighbors stones already placed
    return non_pruned

@njit()
def njit_rollingout(n_turns, engine, all_actions):
    gAction = np.zeros(2, dtype=Typing.TupleDtype)
    turn = 0

    while not engine.isover() and turn < 10: # and turn < n_turns:

        pruning = njit_prunning(engine).flatten().astype(np.bool8)
        if pruning.any():
            actions = all_actions[pruning > 0]
        else:
            actions = all_actions
        
        action_number = len(actions)
        if (action_number == 0):
            with nb.objmode():
                print('rollingout action number 0')
            return
        i = np.random.randint(action_number)
        gAction = actions[i]
        while not engine.is_valid_action(gAction):
            i = np.random.randint(action_number)
            gAction = actions[i]

        engine.apply_action(gAction)
        engine.next_turn()
        turn += 1


@njit()
def test_selection_parallel(actions, policy):
    best_actions = np.zeros((362, 2), dtype=Typing.TupleDtype)

    # action_policy = valid_action(actions, policy)
    action_policy = np.where(actions > 0, policy, 0)
    max = np.amax(action_policy)
    k = 0
    #
    # action_policy = policy * np.where(actions > 0, 1, 0)
    # tmp = np.argwhere(action_policy == np.amax(action_policy))

    for i in range(19):
        for j in range(19):
            if max == action_policy[i, j]:
                best_actions[k][0] = i
                best_actions[k][1] = j
                k += 1

    best_actions[-1, 0] = k
    return best_actions

@njit()
def njit_selection_test(actions, policy, engine):
    gAction = np.zeros(2, dtype=Typing.TupleDtype)
    # action_policy = action_policy.astype(np.float64)
    # c_policy = ffi.cast("double *", action_policy.ctypes.data)
    while True:
        # best_action_count = fastcore.mcts_lazy_selection(c_policy, self.c_best_actions_buffer)
        arr = test_selection_parallel(actions, policy)

        len = arr[-1, 0]
        if (len == 0):
            with nb.objmode():
                print('aled')
            return gAction
            
        arr_pick = np.arange(len)
        # best_actions = np.argwhere(np.amax())
        np.random.shuffle(arr_pick)
        for e in arr_pick:
            x, y = arr[e]
            gAction[:] = (x, y)
            if actions[x, y] == 2:
                return gAction
            elif engine.is_valid_action(gAction):
                actions[x, y] = 2
                return gAction
            else:
                actions[x, y] = 0



@jitclass()
class MCTSWorkerNoJit:
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

    all_actions: nb.types.Array(dtype=nb.int32, ndim=2, layout="C")

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
        self.all_actions = np.empty((361, 2), dtype=np.int32)
        for i in range(19):
            for j in range(19):
                self.all_actions[i * 19 + j, ...] = [np.int32(i), np.int32(j)]
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
            # if (depth > 360):
            #     breakpoint()

            self.end_game = self.engine.isover()
            # statehash = str(self.engine.board.tobytes())
            statehash = self.tobytes(self.engine.board)

        # if (depth > 360):
        #     breakpoint()
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
        return njit_selection_test(state_data.actions, policy, self.engine)
        # best_action = np.zeros(2, dtype=Typing.MCTSIntDtype)
        # available_actions = policy * state_data.actions     # Avaible actions
        # best_actions = np.argwhere(available_actions == np.amax(available_actions))
        # action = best_actions[np.random.randint(len(best_actions))]
        # best_action[0] = action[0]
        # best_action[1] = action[1]
        # # return Typing.nbTuple(best_action)
        # return best_action


    def fill_path(self, depth: Typing.MCTSIntDtype, best_action: np.ndarray):
        self.path[depth].board[...] = self.engine.board
        self.path[depth].player_idx = self.engine.player_idx
        self.path[depth].bestAction[:] = best_action
        
    def expand(self, depth: Typing.MCTSIntDtype):
        self.leaf_data[0].depth = depth
        self.leaf_data[0].visits = 1
        self.leaf_data[0].actions[:] = self.engine.get_actions()
        self.leaf_data[0].heuristic = self.award_end_game() if self.end_game else self.award()


    # def fill_path(self, depth: Typing.MCTSIntDtype, best_action: np.ndarray):
    #     self.path[depth]['board'][...] = self.engine.board
    #     self.path[depth]['player_idx'] = self.engine.player_idx
    #     self.path[depth]['bestAction'][:] = best_action
        
    # def expand(self, depth: Typing.MCTSIntDtype):
    #     self.leaf_data[0]['depth'] = depth
    #     self.leaf_data[0]['visits'] = 1
    #     self.leaf_data[0]['actions'][:] = self.engine.get_actions()
    #     self.leaf_data[0]['heuristic'] = self.award_end_game() if self.end_game else self.award()

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
        # h_leaf = heuristic(self.engine)
        njit_rollingout(10, self.engine, self.all_actions)
        hleaf = 0.5
        if self.engine.isover():
            if self.engine.winner == -1: # DRAW
                h_leaf = 0.5
            else:
                h_leaf = 1 if self.engine.winner == self.engine.player_idx else 0
        # else:
        #     h = heuristic(self.engine)
        #     h_leaf = (h_leaf + h) / 2
        return h_leaf

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
