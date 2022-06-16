from os import stat
import string

import numpy as np

from GomokuLib.Algo import njit_classic_pruning, njit_dynamic_hpruning, njit_heuristic, old_njit_heuristic
import GomokuLib.Typing as Typing
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib.Algo.aligns_graphs import (
    init_my_heuristic_graph,
    init_opp_heuristic_graph,
    init_my_captures_graph,
    init_opp_captures_graph
)

import numba as nb
from numba import njit
from numba.experimental import jitclass
from numba.core.typing import cffi_utils

import fastcore._algo as _fastcore
cffi_utils.register_module(_fastcore)
_algo = _fastcore.lib
ffi = _fastcore.ffi

gettime_ctype = cffi_utils.make_function_type(_algo.gettime)


@nb.vectorize('float32(int32, float32, float32, float32)')
def _get_mc_policy(s_v: np.ndarray, sa_v: np.ndarray, sa_r: np.ndarray,
                     c: Typing.mcts_float_nb_dtype):
    return (sa_r / (sa_v + 1)) + (c * np.sqrt(np.log(s_v) / (sa_v + 1)))


@nb.vectorize('float32(int32, float32, float32, float32, float32, int32, float32)')
def _get_amaf_policy(s_v: np.ndarray, sa_v: np.ndarray, sa_r: np.ndarray,
                     amaf_n: np.ndarray, amaf_v: np.ndarray, amaf_k: Typing.mcts_int_nb_dtype,
                     c: Typing.mcts_float_nb_dtype):
    sa = sa_r / (sa_v + 1)
    amaf = amaf_v / (amaf_n + 1)
    beta = np.sqrt(amaf_k / (amaf_k + 3 * s_v))
    return (beta * amaf + (1 - beta) * sa) + (c * np.sqrt(np.log(s_v) / (sa_v + 1)))


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
    rollingout_turns: Typing.mcts_int_nb_dtype
    amaf_policy: nb.boolean

    states: Typing.nbStateDict
    path: Typing.nbPathArray
    all_actions: Typing.nbAction
    c: Typing.mcts_float_nb_dtype
    get_time_cfunc: gettime_ctype
    amaf_k: Typing.mcts_int_nb_dtype

    depth: Typing.mcts_int_nb_dtype
    max_depth: Typing.mcts_int_nb_dtype
    end_game: nb.boolean
    current_statehash: Typing.nbStrDtype
    gamestatehash: Typing.nbStrDtype

    my_h_graph: Typing.nbHeuristicGraph
    opp_h_graph: Typing.nbHeuristicGraph
    my_cap_graph: Typing.nbHeuristicGraph
    opp_cap_graph: Typing.nbHeuristicGraph
    new_heuristic: nb.boolean
    heuristic_pows: Typing.nbHeuristicData
    heuristic_dirs: Typing.nbHeuristicData
    tmp_h_rewards: Typing.nbHeuristicRewards

    def __init__(self, 
                 engine: Gomoku,
                 iter: Typing.MCTSIntDtype = 1000,
                 rollingout_turns: Typing.MCTSIntDtype = 10,
                 amaf_policy: nb.boolean = False,
                 new_heuristic: nb.boolean = False
                 ):

        print(f"MCTSNjit: __init__(): START")
        self.engine = engine.clone()
        self.mcts_iter = iter
        self.rollingout_turns = rollingout_turns
        self.amaf_policy = amaf_policy
        self.new_heuristic = new_heuristic

        self.c = np.sqrt(2)
        self.amaf_k = np.sqrt(iter)
        self.get_time_cfunc = _algo.gettime

        self.init()
        self.path = np.zeros((361, 2), dtype=Typing.MCTSIntDtype)

        self.all_actions = np.empty((361, 2), dtype=Typing.ActionDtype)
        for i in range(19):
            for j in range(19):
                self.all_actions[i * 19 + j, ...] = [np.int32(i), np.int32(j)]
        # Juste cast à la fin ..

        # Init data for heuristic
        self.my_h_graph = init_my_heuristic_graph()
        self.opp_h_graph = init_opp_heuristic_graph()
        self.my_cap_graph = init_my_captures_graph()
        self.opp_cap_graph = init_opp_captures_graph()
        self.heuristic_pows = np.array([
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            ], dtype=Typing.MCTSIntDtype
        )
        self.heuristic_dirs = np.array([
                [-1, 1],
                [0, 1],
                [1, 1],
                [1, 0]
            ], dtype=Typing.MCTSIntDtype
        )
        self.tmp_h_rewards = np.zeros((21, 21), dtype=Typing.HeuristicGraphDtype)

        # Return a class wrapper to allow player call __call__() and redirect here to do_your_fck_work()
        print(f"MCTSNjit: __init__(): DONE")

    def init(self):
        self.states = nb.typed.Dict.empty(
            key_type=Typing.nbStrDtype,
            value_type=Typing.nbState
        )

    def compile(self, game_engine: Gomoku):
        self.do_your_fck_work(game_engine, 1, 0)

    def str(self):
        return f"MCTSNjit ({self.mcts_iter} iter | newh={1 if self.new_heuristic else 0} | {self.rollingout_turns} roll turns)"

    def get_state_data(self, game_engine: Gomoku) -> Typing.nbStateDict:

        mcts_data = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        statehash = self.fast_tobytes(game_engine.board)
        if statehash in self.states:
            statedata = self.states[statehash]
            mcts_data['mcts_state_data'] = statedata
        else:
            mcts_data['mcts_state_data'] = np.zeros(1, dtype=Typing.StateDataDtype)
        return mcts_data

    def do_your_fck_work(self, game_engine: Gomoku, iter: int, time: int) -> tuple:

        if iter <= 0:
            iter = self.mcts_iter

        self.max_depth = 0
        self.amaf_k = np.sqrt(iter)
        self.gamestatehash = self.fast_tobytes(game_engine.board)

        if time > 0:
            print(f"\n[MCTSNjit: Does {time} ms]\n")

            ts = self.get_time_cfunc()
            time = ts
            while time - ts < time:
                self._do_one_iter(game_engine)
                time = self.get_time_cfunc()

        else:
            print(f"\n[MCTSNjit: Does {iter} iter]\n")

            i = 0
            while i < iter:
                self._do_one_iter(game_engine)
                i += 1

        state_data = self.states[self.gamestatehash][0]
        state_data['max_depth'] = self.max_depth

        sa_v, sa_r = state_data['stateAction']
        arg = np.argmax(sa_r / (sa_v + 1))
        return arg // 19, arg % 19

    def _do_one_iter(self, game_engine: Gomoku):

        self.current_statehash = self.gamestatehash
        self.engine.update(game_engine)

        self.mcts()
        if self.depth + 1 > self.max_depth:
            self.max_depth = self.depth + 1

    def mcts(self):

        old_best_action = np.zeros(2, dtype=Typing.TupleDtype)
        best_action = np.zeros(2, dtype=Typing.TupleDtype)

        # print(f"\n[MCTSNjit mcts function iter {mcts_iter}]\n")
        self.depth = 0
        self.end_game = self.engine.isover()
        statehashes = []
        while self.current_statehash in self.states and not self.end_game:

            state_data = self.states[self.current_statehash][0]

            policy = self.get_policy(state_data)
            pruning = self.dynamic_pruning(state_data['pruning'])

            old_best_action[:] = best_action
            best_action = self.lazy_selection(policy * pruning, state_data['actions'])

            statehashes.append(self.current_statehash)
            self.path[self.depth][:] = best_action
            rawidx = best_action[0] * 19 + best_action[1]
            newStone = '1' if self.engine.player_idx == 0 else '2'
            self.current_statehash = self.current_statehash[:rawidx] + newStone + self.current_statehash[rawidx + 1:]

            self.engine.apply_action(best_action)
            self.engine.next_turn()
            self.depth += 1

            # remove captured stone from statehash
            if self.engine.is_capture_active:
                captures = self.engine.capture.captured_buffer
                for i in range(self.engine.capture.capture_count):
                    rawidx0 = captures[i, 0, 0] * 19 + captures[i, 0, 1]
                    rawidx1 = captures[i, 1, 0] * 19 + captures[i, 1, 1]
                    if rawidx0 > rawidx1:
                        rawidx1 ^= rawidx0
                        rawidx0 ^= rawidx1
                        rawidx1 ^= rawidx0
                    self.current_statehash = (
                        self.current_statehash[:rawidx0] + '0'
                        + self.current_statehash[rawidx0 + 1:rawidx1] + '0'
                        + self.current_statehash[rawidx1 + 1:]
                    )
            #####

            self.end_game = self.engine.isover()

        # Some tasks need to be done before award, because rollingout transforms self.engine
        actions = self.engine.get_lazy_actions()
        pruning_arr = self.new_state_pruning()

        # After all engine data fetching
        if self.depth > 1:
            old_statehash = statehashes[-2]
        else:
            old_statehash = None
        reward = self.award(old_statehash, best_action, old_best_action)

        self.expand(actions, reward, pruning_arr)
        self.backpropagation(statehashes, reward)

    def get_policy(self, state_data: Typing.StateDataDtype) -> Typing.nbPolicy:
        """
            ucb(s, a) = exploitation_rate(s, a) + exploration_rate(s, a)

            exploitation_rate(s, a):
                AMAFQuality(s, a) = beta * AMAF(s, a) + (1 - beta) * quality(s, a)
                    
                    beta = sqrt(amaf_k / (amaf_k + 3 * visits(s)))
                        amaf_k: number of simulations at which the Monte-Carlo value and the
                                AMAF value should be given equal weigh (beta=0.5)
                        0 < beta < 1

                    quality(s, a) = rewards(s, a)     / (visits(s, a)     + 1)
                    AMAF(s, a)    = rewardsAMAF(s, a) / (visitsAMAF(s, a) + 1)
                
            exploration_rate(s, a) =    c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )

        """
        sa_v, sa_r = state_data['stateAction']

        if self.amaf_policy:
            amaf_n, amaf_v = state_data['amaf']
            return _get_amaf_policy(state_data['visits'], sa_v, sa_r, amaf_n, amaf_v, self.amaf_k, self.c)
        else:
            return _get_mc_policy(state_data['visits'], sa_v, sa_r, self.c)

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
            if (len == 0):
                return gAction
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

    def expand(self, actions: np.ndarray, reward: Typing.heuristic_graph_nb_dtype, pruning_arr: np.ndarray):
        state = np.zeros(1, dtype=Typing.StateDataDtype)

        state[0]['max_depth'] = self.depth
        state[0]['visits'] = 1
        state[0]['rewards'] = reward       # Useless data for MCTS, usefull for UI
        state[0]['stateAction'][...] = 0.
        state[0]['actions'][...] = actions
        state[0]['heuristic'] = reward
        state[0]['pruning'][...] = pruning_arr
        state[0]['amaf'][...] = 0.

        state[0]['h_rewards'][...] = self.tmp_h_rewards
        state[0]['h_captures'][...] = self.engine.capture.get_captures()

        self.states[self.current_statehash] = state

    def new_state_pruning(self, engine: Gomoku = None):

        if engine is None:
            engine = self.engine

        game_zone = engine.get_game_zone()
        g0 = game_zone[0]
        g1 = game_zone[1]
        g2 = game_zone[2]
        g3 = game_zone[3]
        return njit_dynamic_hpruning(engine.board, g0, g1, g2, g3, engine.player_idx,
            self.my_h_graph, self.opp_h_graph, self.my_cap_graph, self.opp_cap_graph)

    def dynamic_pruning(self, pruning_arr: np.ndarray):
        if self.depth == 0:        # Depth 0
            return pruning_arr[0]
        if self.depth > 2:         # Depth 3 | ...
            return pruning_arr[2]
        else:                      # Depth 1 | 2
            return pruning_arr[1]

    def classic_pruning(self):
        return njit_classic_pruning(self.engine.board)

    def award(self, old_statehash, best_action, old_best_action):

        if self.engine.isover():
            return 1 if self.engine.winner == self.engine.player_idx else 0
        else:
            if self.new_heuristic:  # Func ptr set in __init__
                return self.dynamic_heuristic(old_statehash, best_action, old_best_action)
            else:
                return self.heuristic()

            # dh = self.dynamic_heuristic(old_statehash, best_action, old_best_action)
            # h = self.heuristic()
            # if h != dh:
            #
            #     print("depth ", self.depth, " dh != h:", dh, h)
            #     # print(self.tmp_h_rewards)
            #     old_state_data = self.states[old_statehash][0]
            #     old_c0 = old_state_data['h_captures'][self.engine.player_idx]        # Same player_idx
            #     old_c1 = old_state_data['h_captures'][self.engine.player_idx ^ 1]
            #     print(self.tmp_h_rewards)
            #
            #     print(old_c0, old_c1)
            #     print(self.engine.get_captures())
            #     # breakpoint()
            #
            # return h

    def dynamic_heuristic(self, old_statehash, best_action, old_best_action):
        board = self.engine.board

        cap = self.engine.get_captures()
        c0 = cap[self.engine.player_idx]
        c1 = cap[self.engine.player_idx ^ 1]

        game_zone = self.engine.get_game_zone()
        g0 = game_zone[0]
        g1 = game_zone[1]
        g2 = game_zone[2]
        g3 = game_zone[3]

        if self.depth > 1:
            old_state_data = self.states[old_statehash][0]
            self.tmp_h_rewards[...] = old_state_data['h_rewards']    # Data from 2 turns.
            old_c0 = old_state_data['h_captures'][self.engine.player_idx]        # Same player_idx
            old_c1 = old_state_data['h_captures'][self.engine.player_idx ^ 1]
        else:
            self.tmp_h_rewards = np.zeros((21, 21), dtype=Typing.HeuristicGraphDtype)
            old_c0 = 0
            old_c1 = 0

        return njit_heuristic(board, c0, c1, g0, g1, g2, g3, self.engine.player_idx,
            self.my_h_graph, self.opp_h_graph, self.my_cap_graph, self.opp_cap_graph, self.heuristic_pows, self.heuristic_dirs,
            self.tmp_h_rewards, best_action[0], best_action[1], old_best_action[0], old_best_action[1], old_c0, old_c1)

    def heuristic(self, engine: Gomoku = None):
        if engine is None:
            engine = self.engine

        board = engine.board

        cap = engine.get_captures()
        c0 = cap[engine.player_idx]
        c1 = cap[engine.player_idx ^ 1]

        game_zone = engine.get_game_zone()
        g0 = game_zone[0]
        g1 = game_zone[1]
        g2 = game_zone[2]
        g3 = game_zone[3]

        return old_njit_heuristic(board, c0, c1, g0, g1, g2, g3, engine.player_idx,
            self.my_h_graph, self.opp_h_graph, self.my_cap_graph, self.opp_cap_graph, self.heuristic_pows, self.heuristic_dirs)

    def rollingout(self):
        turn = 0
        while turn < self.rollingout_turns and not self.engine.isover():

            pruning = self.classic_pruning()
            pruning = pruning.flatten().astype(np.bool8)

            # Create actions from pruning
            if pruning.any():
                actions = self.all_actions[pruning > 0]
            else:
                actions = self.all_actions

            # Select randomly an action from actions/pruning
            action_number = len(actions)

            arr = np.arange(action_number)
            np.random.shuffle(arr)
            # i = np.random.randint(action_number)
            i = 0
            gAction = actions[arr[0]]
            while (i < action_number and not self.engine.is_valid_action(gAction)):
                gAction = actions[arr[i]]
                i += 1

            if (i == action_number):
                return
            self.engine.apply_action(gAction)
            self.engine.next_turn()
            turn += 1

    def backpropagation(self, statehashes, reward: Typing.heuristic_graph_nb_dtype):

        # Init backprop AMAF buffer
        amaf_masks = np.zeros((2, 2, 19, 19))    # sAMAF_v and sAMAF_n for 2 players
        
        p_id = 0
        for i in range(self.depth - 1, -1, -1):
            # Flip data
            reward = 1 - reward
            p_id ^= 1
            # reward = 1 - (0.90 * reward)

            # print("Backprop ", i, " reward ", reward)
            self.backprop_memory(self.path[i], reward, statehashes[i], p_id, amaf_masks)

    def backprop_memory(self, best_action, reward: Typing.heuristic_graph_nb_dtype,
        statehash: string, p_id: int, amaf_masks: np.ndarray):
        stateAction_update = np.ones(2, dtype=Typing.MCTSFloatDtype)
        stateAction_update[1] = reward

        r, c = best_action
        state_data = self.states[statehash][0]

        state_data['visits'] += 1                           # update n count
        state_data['rewards'] += reward                     # Useless data for MCTS, usefull for UI
        state_data['stateAction'][..., r, c] += stateAction_update  # update state-action count / value

        amaf_masks[p_id, ..., r, c] += stateAction_update  # update state-action count / value
        state_data['amaf'] += amaf_masks[p_id]

    def fast_tobytes(self, arr: Typing.BoardDtype):
        byte_list = []
        for i in range(19):
            for j in range(19):
                byte_list.append('1' if arr[0, i, j] == 1 else ('2' if arr[1, i, j] == 1 else '0'))
        return ''.join(byte_list)
