import numpy as np

from GomokuLib.Algo import my_heuristic_graph, opp_heuristic_graph, njit_heuristic
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



@nb.vectorize('float64(int8, float64)')
def _valid_policy_action(actions, policy):
    if actions > 0:
        return policy
    else:
        return 0


@jitclass()
class MCTSNjit:

    engine: Gomoku
    mcts_iter: Typing.mcts_int_nb_dtype
    is_pruning: nb.boolean
    rollingout_turns: Typing.mcts_int_nb_dtype
    with_new_heuristic: nb.boolean

    states: Typing.nbStateDict
    path: Typing.nbPath
    all_actions: Typing.nbAction
    c: Typing.mcts_float_nb_dtype
    my_heuristic_graph: Typing.nbHeuristicGraph
    opp_heuristic_graph: Typing.nbHeuristicGraph

    depth: Typing.mcts_int_nb_dtype
    end_game: nb.boolean
    reward: Typing.mcts_float_nb_dtype

    def __init__(self, 
                 engine: Gomoku,
                 iter: Typing.MCTSIntDtype = 1000,
                 pruning: nb.boolean = True,
                 rollingout_turns: Typing.MCTSIntDtype = 10,
                 with_new_heuristic: nb.boolean = True
                 ):

        self.engine = engine.clone()
        self.mcts_iter = iter
        self.is_pruning = pruning
        self.rollingout_turns = rollingout_turns
        self.with_new_heuristic = with_new_heuristic
        self.c = np.sqrt(2)

        self.states = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        self.path = np.zeros(361, dtype=Typing.PathDtype)

        self.all_actions = np.empty((361, 2), dtype=Typing.ActionDtype)
        for i in range(19):
            for j in range(19):
                self.all_actions[i * 19 + j, ...] = [np.int32(i), np.int32(j)]

        # self.my_heuristic_graph = init_my_heuristic_graph()
        # self.opp_heuristic_graph = init_opp_heuristic_graph()

        print(f"{self.str()}: end __init__()\n")
        # Return a class wrapper to allow player call __call__() and redirect here to do_your_fck_work()

    def str(self):
        return f"MCTSNjit ({self.mcts_iter} iter) newh={1 if self.with_new_heuristic else 0}"

    def get_state_data(self, game_engine: Gomoku) -> Typing.nbStateDict:

        mcts_data = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        statehash = self.tobytes(game_engine.board)
        if statehash in self.states:
            mcts_data['mcts_state_data'] = self.states[statehash]
        else:
            mcts_data['mcts_state_data'] = np.zeros(1, dtype=Typing.StateDataDtype)
        return mcts_data
        # return {
        #     'mcts_state_data': self.states[self.tobytes(game_engine.board)][0],
        # }

    def get_state_data_after_action(self, game_engine: Gomoku):
        statehash = self.tobytes(game_engine.board)
        if statehash in self.states:
            h = self.states[statehash][0]['heuristic']
        else:
            h = self.heuristic(game_engine, debug=True)
        
        return {
            'heuristic': h
        }

    def do_your_fck_work(self, game_engine: Gomoku) -> tuple:

        print(f"\n[MCTSNjit __call__() for {self.mcts_iter} iter]\n")
        self.do_n_iter(game_engine, self.mcts_iter)

        gamestatehash = self.tobytes(game_engine.board)
        state_data = self.states[gamestatehash][0]
        sa_v, sa_r = state_data['stateAction']
        arg = np.argmax(sa_r / (sa_v + 1))
        # print(f"StateAction visits: {sa_v}")
        # print(f"StateAction reward: {sa_r}")
        # print(f"Qualities: {sa_r / sa_v}")

        # print(f"argmax: {arg} / {int(arg / 19)} {arg % 19}")
        return int(arg / 19), arg % 19

    def do_n_iter(self, game_engine: Gomoku, iter: int):
        for i in range(iter):
            self.engine.update(game_engine)
            self.mcts(i)

    def mcts(self, mcts_iter: Typing.MCTSIntDtype):

        # print(f"\n[MCTSNjit mcts function iter {mcts_iter}]\n")
        self.depth = 0
        self.end_game = self.engine.isover()
        statehash = self.tobytes(self.engine.board)
        while statehash in self.states and not self.end_game:

            state_data = self.states[statehash][0]
            # print(f"MCTS_iter={mcts_iter} | depth={self.depth} | statehash={statehash}")

            policy = self.get_policy(state_data, self.c)
            # best_action = self.selection(policy, state_data)
            best_action = self.lazy_selection(policy, state_data['actions'])

            # with nb.objmode():
            #     print(f"best_action {best_action[0]} {best_action[1]} | self.states[statehash][0]['stateAction']:\n{self.states[statehash][0]['stateAction']}")
            #     # print(f"policy:\n{policy}")
            #     # print(f"best_action:\n{best_action}")
            #     breakpoint()
            #     pass

            if self.depth > 50:
                print(f"Depth > 50")
                with nb.objmode():
                    breakpoint()

            self.fill_path(statehash, best_action)
            self.engine.apply_action(best_action)
            self.engine.next_turn()
            self.depth += 1

            self.end_game = self.engine.isover()
            statehash = self.tobytes(self.engine.board)

        # self.fill_path(statehash, np.full(2, -1, Typing.MCTSIntDtype))
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
        ucbs = sa_r / (sa_v + 1) + self.c * np.sqrt(np.log(s_v) / (sa_v + 1))
        return ucbs * state_data['pruning']

    def get_best_policy_actions(self, policy: np.ndarray, actions: Typing.ActionDtype):
        best_actions = np.zeros((362, 2), dtype=Typing.TupleDtype)

        # action_policy = np.where(actions > 0, policy, 0) # Replace by @nb.vectorize ?
        action_policy = _valid_policy_action(actions, policy)

        pmax = np.amax(action_policy)
        k = 0
        for i in range(19):
            for j in range(19):
                if pmax == action_policy[i, j]:
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
            # if (len == 0):
            #     with nb.objmode():
            #         print('aled lazy selection')
            #     return gAction

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

    def fill_path(self, statehash, best_action: np.ndarray):
        # print(f"Fill path depth {self.depth} statehash:\n{statehash}")

        # sh = np.unicode_(statehash)
        # print(f"Statehash : {sh} {sh.dtype}")

        self.path[self.depth]['board'][...] = self.engine.board
        # self.path[self.depth]['statehash'] = statehash
        self.path[self.depth]['player_idx'] = self.engine.player_idx
        self.path[self.depth]['bestAction'][:] = best_action
        # with nb.objmode():
        #     print(f"Fill path depth {self.depth} statehash:\n{self.path[self.depth]['statehash']}")

    def expand(self, statehash: str):
        # print(f"Expand depth {self.depth} statehash:\n{statehash}")
        self.states[statehash] = np.zeros(1, dtype=Typing.StateDataDtype)

        actions = self.engine.get_lazy_actions()
        # actions = self.engine.get_actions()
        pruning = self.pruning()
        self.reward = self.award_end_game() if self.end_game else self.award()  # After all engine data fetching

        self.states[statehash][0]['depth'] = self.depth
        self.states[statehash][0]['visits'] = 1
        self.states[statehash][0]['rewards'] = self.reward
        self.states[statehash][0]['stateAction'][...] = 0.
        self.states[statehash][0]['actions'][...] = actions
        self.states[statehash][0]['heuristic'] = self.reward
        self.states[statehash][0]['pruning'][...] = pruning
    
        # with nb.objmode():
        #     print(f"Expand:\n{self.states[statehash]}")

    def award_end_game(self):
        if self.engine.winner == -1: # Draw
            return 0.5
        return 1 if self.engine.winner == self.engine.player_idx else 0

    # def award(self):
    #     """
    #         Mean of leaf state heuristic & random(pruning) rollingout end state heuristic
    #     """
    #     self.rollingout()

    #     if self.engine.isover():
    #         return self.award_end_game()
    #     else:
    #         return 0.5

    def award(self):
        """
            Mean of leaf state heuristic & random(pruning) rollingout end state heuristic
        """
        h_leaf = self.heuristic()
        if self.rollingout_turns:
            self.rollingout()
    
            if self.engine.isover():
                return self.award_end_game()
            else:
                h = self.heuristic()
                return (h_leaf + (1 - h if self.rollingout_turns % 2 else h)) / 2
        else:
            return h_leaf

    def heuristic(self, engine: Gomoku = None, debug=False):

        if engine is None:
            engine = self.engine

        board = engine.board

        cap = engine.get_captures()
        c0 = cap[0]
        c1 = cap[1]

        game_zone = engine.get_game_zone()
        g0 = game_zone[0]
        g1 = game_zone[1]
        g2 = game_zone[2]
        g3 = game_zone[3]

        if self.with_new_heuristic:
            h = njit_heuristic(board, my_heuristic_graph, opp_heuristic_graph, c0, c1, g0, g1, g2, g3)

        else:
            c_board = ffi.from_buffer(board)
            c_full_board = ffi.from_buffer(board[0] | board[1])

            h = _algo.mcts_eval_heuristic(
                c_board, c_full_board,
                c0, c1, g0, g1, g2, g3
            )

        # if self.with_new_heuristic and debug:
        #     print(board)
        #     print(cap)
        #     print(game_zone)
        #     print(self.with_new_heuristic)
        #     print(float(h))
        #     # breakpoint()

        return h

    def rollingout(self):
        # gAction = np.zeros(2, dtype=Typing.TupleDtype)
        turn = 0
        while not self.engine.isover() and turn < self.rollingout_turns:
        # while not self.engine.isover():

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
                    breakpoint()
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

        neigh = np.zeros((19, 19), dtype=Typing.PruningDtype)

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

        if not self.is_pruning:
            return np.ones((19, 19), dtype=Typing.PruningDtype)

        full_board = (self.engine.board[0] | self.engine.board[1]).astype(Typing.PruningDtype)
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

        # Start with the last play (Penultimate state/board)
        reward = 1 - self.reward
        for i in range(self.depth - 1, -1, -1):
            # print(f"Backprop path index {i}")
            self.backprop_memory(self.path[i], reward)
            reward = 1 - reward

    def backprop_memory(self, memory: Typing.StateDataDtype, reward: Typing.MCTSFloatDtype):
        # print(f"Memory:\n{memory}")
        
        board = memory['board']
        r, c = memory['bestAction']
        # statehash = memory['statehash']
        # print(f"memory['bestAction']:\n{r} {c}")

        # if statehash not in self.states:
        #     print(f"statehash not in states !!! ->\n{statehash}")
        #     breakpoint()
        #     pass
        state_data = self.states[self.tobytes(board)][0]
        # state_data = self.states[statehash][0]

        state_data['visits'] += 1                           # update n count
        state_data['rewards'] += reward                     # update state value / Use for ?...

        stateAction_update = np.ones(2, dtype=Typing.MCTSFloatDtype)
        stateAction_update[1] = reward
        state_data['stateAction'][..., r, c] += stateAction_update  # update state-action count / value

    def tobytes(self, arr: Typing.BoardDtype):
        return ''.join(map(str, map(np.int8, np.nditer(arr)))) # Aled la ligne (nogil parallele mieux ?..)
        # return np.char.join('', map(np.unicode_, np.nditer(arr)))
