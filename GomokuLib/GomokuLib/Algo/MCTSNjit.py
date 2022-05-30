import GomokuLib

import numpy as np

import GomokuLib.Typing as Typing
from GomokuLib.Game.GameEngine import Gomoku

import numba as nb
from numba import njit
from numba.experimental import jitclass
from numba.core.typing import cffi_utils
import fastcore._algo as _fastcore

cffi_utils.register_module(_fastcore)
_algo = _fastcore.lib
ffi = _fastcore.ffi



# @numba.vectorize('float64(int8, float64)')
# def valid_action(actions, policy):
#     if actions > 0:
#         return policy
#     else:
#         return 0

@jitclass()
class MCTSNjit:

    engine: Gomoku
    mcts_iter: Typing.mcts_int_nb_dtype
    is_pruning: nb.boolean
    rollingout_turns: Typing.mcts_int_nb_dtype

    states: Typing.nbStateDict
    leaf_data: Typing.nbState
    path: Typing.nbPath
    all_actions: Typing.nbAction
    c: Typing.mcts_float_nb_dtype

    depth: Typing.mcts_int_nb_dtype
    end_game: nb.boolean
    reward: Typing.mcts_float_nb_dtype

    def __init__(self, 
                 engine: Gomoku,
                 iter: Typing.MCTSIntDtype = 1000,
                 pruning: nb.boolean = True,
                 rollingout_turns: Typing.MCTSIntDtype = 10
                 ):

        self.engine = engine.clone()
        self.mcts_iter = iter
        self.is_pruning = pruning
        self.rollingout_turns = rollingout_turns
        self.c = np.sqrt(2)

        self.states = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        # self.leaf_data = np.recarray(1, dtype=Typing.StateDataDtype)
        # self.path = np.recarray(361, dtype=Typing.PathDtype)
        # self.leaf_data[...] = np.zeros(1, dtype=Typing.StateDataDtype)
        # self.path[...] = np.zeros(361, dtype=Typing.PathDtype)

        self.leaf_data = np.zeros(1, dtype=Typing.StateDataDtype)
        self.path = np.zeros(361, dtype=Typing.PathDtype)

        self.all_actions = np.empty((361, 2), dtype=Typing.ActionDtype)
        for i in range(19):
            for j in range(19):
                self.all_actions[i * 19 + j, ...] = [np.int32(i), np.int32(j)]

        print(f"{self.__str__()}: end __init__()\n")
        # Return a class wrapper to allow player call __call__() and redirect here to do_your_fck_work()

    def __str__(self):
        return f"MCTSNjit iter={self.mcts_iter}"

    def get_state_data(self, engine: Gomoku):
        return {
            'mcts_state_data': self.states[self.tobytes(engine.board)][0],
        }

    def get_state_data_after_action(self, engine: Gomoku):
        statehash = self.tobytes(engine.board)
        return {
            'heuristic': self.states[statehash][0]['heuristic'] if statehash in self.states else 0.5
        }

    def do_your_fck_work(self, game_engine: Gomoku) -> tuple:

        print(f"\n[MCTSNjit __call__()]\n")
        for mcts_iter in range(self.mcts_iter):
            self.engine.update(game_engine)
            self.mcts(mcts_iter)

        gamestatehash = self.tobytes(game_engine.board)
        state_data = self.states[gamestatehash][0]
        sa_v, sa_r = state_data['stateAction']
        sa_v += 1
        arg = np.argmax(sa_r / sa_v)
        # print(f"StateAction visits: {sa_v}")
        # print(f"StateAction reward: {sa_r}")
        # print(f"Qualities: {sa_r / sa_v}")
        print(f"argmax: {arg} / {int(arg / 19)} {arg % 19}")
        return int(arg / 19), arg % 19

    def mcts(self, mcts_iter: Typing.MCTSIntDtype):

        # print(f"\n[MCTSNjit mcts function iter {mcts_iter}]\n")
        self.depth = 0
        self.end_game = self.engine.isover()
        statehash = self.tobytes(self.engine.board)
        while statehash in self.states and not self.end_game:

            state_data = self.states[statehash][0]

            policy = self.get_policy(state_data, self.c)
            # best_action = self.selection(policy, state_data)
            best_action = self.lazy_selection(policy, state_data['actions'])

            # print(f"MCTS_iter={mcts_iter} | depth={self.depth} | statehash in self.states={'True' if statehash in self.states else 'False'} | actions {best_action[0]} {best_action[1]}")
            # if any([np.all(best_action == p['bestAction']) for p in self.path[:self.depth+1]]):
            #     # with nb.objmode():
            #     #     print(f"state_data: {state_data}")
            #     #     pass # Same action
            #
            #     print(f"self.path[:self.depth]['bestAction'] ->\n{self.path[:self.depth+1]['bestAction']}")
            #     breakpoint()

            # if self.depth > 20:
            #     print(f"Depth > 20")
            #     with nb.objmode():
            #         breakpoint()

            self.fill_path(best_action)
            self.engine.apply_action(best_action)
            self.engine.next_turn()
            self.depth += 1

            self.end_game = self.engine.isover()
            statehash = self.tobytes(self.engine.board)

        self.fill_path(np.full(2, -1, Typing.MCTSIntDtype))
        self.expand(statehash)
        self.backpropagation()

    def get_policy(self, state_data: Typing.nbState, *args) -> Typing.nbPolicy:
        """
            ucb(s, a) = exploitation_rate(s, a) + exploration_rate(s, a)

            exploitation_rate(s, a) =   reward(s, a) / (visits(s, a) + 1)
            exploration_rate(s, a) =    c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )

        """
        s_v = state_data['visits']
        sa_v, sa_r = state_data['stateAction']
        sa_v += 1   # Init this value at 1 ?
        ucbs = sa_r / sa_v + self.c * np.sqrt(np.log(s_v) / sa_v)
        # if self.is_pruning:
        #     return ucbs * state_data.pruning
        return ucbs

    def get_best_policy_actions(self, policy: np.ndarray, actions: Typing.ActionDtype):
        best_actions = np.zeros((362, 2), dtype=Typing.TupleDtype)

        action_policy = np.where(actions > 0, policy, 0) # Replace by @nb.vectorize ?
        max = np.amax(action_policy)
        k = 0
        for i in range(19):
            for j in range(19):
                if max == action_policy[i, j]:
                    best_actions[k][0] = i
                    best_actions[k][1] = j
                    k += 1

        best_actions[-1, 0] = k
        return best_actions

    def lazy_selection(self, policy: np.ndarray, actions: Typing.ActionDtype):
        gAction = np.zeros(2, dtype=Typing.TupleDtype)

        while True:
            arr = self.get_best_policy_actions(policy, actions)

            len = arr[-1, 0]
            if (len == 0):
                with nb.objmode():
                    print('aled lazy selection')
                return gAction

            arr_pick = np.arange(len)
            np.random.shuffle(arr_pick) # Slow ...
            for e in arr_pick:
                x, y = arr[e]
                gAction[:] = (x, y)
                if actions[x, y] == 2:
                    return gAction
                elif self.engine.is_valid_action(gAction):
                    actions[x, y] = 2
                    return gAction
                else:
                    actions[x, y] = 0

    # def selection(self, policy: np.ndarray, state_data: Typing.nbState) -> Typing.nbTuple:
    #     best_action = np.zeros(2, dtype=Typing.MCTSIntDtype)
    #     available_actions = policy * state_data['actions']     # Avaible actions
    #     # available_actions = policy * state_data.actions     # Avaible actions
        
    #     best_actions = np.argwhere(available_actions == np.amax(available_actions))
        
    #     action = best_actions[np.random.randint(len(best_actions))]
    #     best_action[0] = action[0]
    #     best_action[1] = action[1]
    #     # return Typing.nbTuple(best_action)
    #     breakpoint()
    #     return best_action

    def fill_path(self, best_action: np.ndarray):
        self.path[self.depth]['board'][...] = self.engine.board
        self.path[self.depth]['player_idx'] = self.engine.player_idx
        self.path[self.depth]['bestAction'][:] = best_action

    def expand(self, statehash: str):
        self.states[statehash] = np.zeros(1, dtype=Typing.StateDataDtype)

        actions = self.engine.get_actions()
        self.reward = self.award_end_game() if self.end_game else self.award()

        self.states[statehash][0]['depth'] = self.depth
        self.states[statehash][0]['visits'] = 1
        self.states[statehash][0]['actions'][:] = actions
        self.states[statehash][0]['heuristic'] = self.reward
        # if self.is_pruning:
        #     self.states[statehash][0].pruning = pruning(self.engine)

    def award_end_game(self):
        if self.engine.winner == -1: # Draw
            return 0.5
        return 1 if self.engine.winner == self.engine.player_idx else 0

    def award(self):
        """
            Mean of leaf state heuristic & random(pruning) rollingout end state heuristic
        """
        # self.rollingout()

        if self.engine.isover():
            return self.award_end_game()
        else:
            return 0.5
    # def award(self):
    #     """
    #         Mean of leaf state heuristic & random(pruning) rollingout end state heuristic
    #     """
    #     h_leaf = self.heuristic()
    #     if self.rollingout_turns:
    #         self.rollingout()
    #
    #         if self.engine.isover():
    #             return self.award_end_game()
    #         else:
    #             h = self.heuristic()
    #             return (h_leaf + (1 - h if self.rollingout_turns % 2 else h)) / 2
    #     else:
    #         return h_leaf

    # def heuristic(self):
    #     board = self.engine.board

    #     c_board = ffi.from_buffer(board)
    #     c_full_board = ffi.from_buffer(board[0] | board[1])

    #     cap = self.engine.get_captures()
    #     c0 = cap[0]
    #     c1 = cap[1]

    #     game_zone = self.engine.get_game_zone()
    #     g0 = game_zone[0]
    #     g1 = game_zone[1]
    #     g2 = game_zone[2]
    #     g3 = game_zone[3]

    #     x = _algo.mcts_eval_heuristic(
    #         c_board, c_full_board,
    #         c0, c1, g0, g1, g2, g3
    #     )
    #     return x

    def rollingout(self):
        # gAction = np.zeros(2, dtype=Typing.TupleDtype)
        turn = 0
        # while not self.engine.isover() and turn < self.rollingout_turns:
        while not self.engine.isover():

            pruning = self.pruning().flatten().astype(np.bool8)

            # Create actions from pruning
            if pruning.any():
                actions = self.all_actions[pruning > 0]
            else:
                actions = self.all_actions

            # Select randomly an action from actions/pruning
            action_number = len(actions)

            if (action_number == 0):
                with nb.objmode():
                    print('rollingout action number 0')
                return

            i = np.random.randint(action_number)
            gAction = actions[i]
            while not self.engine.is_valid_action(gAction):
                i = np.random.randint(action_number)
                gAction = actions[i]

            self.engine.apply_action(gAction)
            self.engine.next_turn()
            turn += 1

    def get_neighbors_mask(self, board):

        neigh = np.zeros((19, 19), dtype=board.dtype)

        neigh[:-1, :] |= board[1:, :]  # Roll cols to left
        neigh[1:, :] |= board[:-1, :]  # Roll cols to right
        neigh[:, :-1] |= board[:, 1:]  # Roll rows to top
        neigh[:, 1:] |= board[:, :-1]  # Roll rows to bottom

        neigh[1:, 1:] |= board[:-1, :-1]  # Roll cells to the right-bottom corner
        neigh[1:, :-1] |= board[:-1, 1:]  # Roll cells to the right-upper corner
        neigh[:-1, 1:] |= board[1:, :-1]  # Roll cells to the left-bottom corner
        neigh[:-1, :-1] |= board[1:, 1:]  # Roll cells to the left-upper corner

        return neigh

    def pruning(self):

        full_board = (self.engine.board[0] | self.engine.board[1]).astype(np.bool8)
        non_pruned = self.get_neighbors_mask(full_board)  # Get neightbors, depth=1

        # if hard_pruning:
        #     non_pruned = n1
        # else:
        #     n2 = self.get_neighbors_mask(n1)  # Get neightbors, depth=2
        #     non_pruned = np.logical_or(n1, n2)

        xp = non_pruned ^ full_board
        non_pruned = xp & non_pruned  # Remove neighbors stones already placed
        return non_pruned

    def backpropagation(self):

        reward = self.reward
        for i in range(self.depth, -1, -1):
            self.backprop_memory(self.path[i], reward)
            reward = 1 - reward

    def backprop_memory(self, memory: Typing.StateDataDtype, reward: Typing.MCTSFloatDtype):
        # print(f"Memory:\n{memory}")
        # print(f"Memory dtype:\n{memory.dtype}")
        board = memory['board']
        bestAction = memory['bestAction']

        state_data = self.states[self.tobytes(board)][0]

        state_data['visits'] += 1                           # update n count
        state_data['rewards'] += reward                     # update state value
        if bestAction[0] == -1:
            return

        r, c = bestAction
        stateAction_update = np.ones(2, dtype=Typing.MCTSFloatDtype)
        stateAction_update[1] = reward
        state_data['stateAction'][..., r, c] += stateAction_update  # update state-action count / value
        # breakpoint()

    def tobytes(self, arr: Typing.nbBoard):
        return ''.join(map(str, map(np.int8, np.nditer(arr))))
