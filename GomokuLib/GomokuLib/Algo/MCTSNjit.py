import GomokuLib

import numpy as np
import numba as nb

from numba.experimental import jitclass

import GomokuLib.Typing as Typing
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib.Algo.MCTSUtils import MCTSUtils




@jitclass()
class MCTSNjit:

    engine: Gomoku
    mcts_iter: Typing.mcts_int_nb_dtype
    pruning: nb.boolean
    rollingout_turns: Typing.mcts_int_nb_dtype
    c: Typing.mcts_float_nb_dtype

    states: Typing.nbStateDict
    leaf_data: Typing.nbState
    path: Typing.nbPath

    depth: Typing.mcts_int_nb_dtype
    end_game: nb.boolean

    def __init__(self, 
                 engine: Gomoku,
                 iter: Typing.MCTSIntDtype = 1000,
                 pruning: nb.boolean = True,
                 rollingout_turns: Typing.MCTSIntDtype = 10
                 ):

        self.engine = engine.clone()
        self.mcts_iter = iter
        self.pruning = pruning
        self.rollingout_turns = rollingout_turns
        self.c = np.sqrt(2)

        self.states = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=Typing.nbState
        )
        self.leaf_data = np.zeros(1, dtype=Typing.StateDataDtype)
        self.path = np.zeros(361, dtype=Typing.PathDtype)

        print(f"{self.__str__()}: end __init__()\n")
        # Return a class wrapper to allow player call __call__() and redirect here to do_your_fck_work()

    def __str__(self):
        return f"MCTSNjit iter={self.mcts_iter}"

    def do_your_fck_work(self, game_engine: Gomoku) -> tuple:

        for mcts_iter in range(self.mcts_iter):
            self.engine.update(game_engine)
            self.mcts(mcts_iter)

        gamestatehash = self.tobytes(self.game_engine.board)
        state_data = self.states[gamestatehash][0]
        sa_v, sa_r = state_data.stateAction
        sa_v += 1
        arg = np.argmax(sa_r / sa_v)
        # print(f"StateAction visits: {sa_v}")
        # print(f"StateAction reward: {sa_r}")
        # print(f"Qualities: {sa_r / sa_v}")
        print(f"argmax: {arg} / {int(arg / 19)} {arg % 19}")
        return int(arg / 19), arg % 19

    def mcts(self, mcts_iter: Typing.MCTSIntDtype):

        # print(f"\n[MCTSNjit function iter {mcts_iter}]\n")
        self.depth = 0
        self.end_game = self.engine.isover()
        statehash = self.tobytes(self.engine.board)
        while statehash in self.states and not self.end_game:

            state_data = self.states[statehash][0]

            policy = self.get_policy(state_data, self.c)
            best_action = self.selection(policy, state_data)

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

    def fill_path(self, best_action: np.ndarray):
        self.path[self.depth].board[...] = self.engine.board
        self.path[self.depth].player_idx = self.engine.player_idx
        self.path[self.depth].bestAction[:] = best_action
        
    def expand(self, statehash: str):
        self.states[statehash][0].depth = self.depth
        self.states[statehash][0].visits = 1
        self.states[statehash][0].actions[:] = self.engine.get_actions()
        self.states[statehash][0].heuristic = self.award_end_game() if self.end_game else self.award()
        # if self.pruning:
        #     self.states[statehash][0].pruning = MCTSUtils.pruning(self.engine)

    def award_end_game(self):
        if self.engine.winner == -1: # Draw
            return 0.5
        return 1 if self.engine.winner == self.engine.player_idx else 0

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

    def backpropagation(self, reward: Typing.MCTSFloatDtype):

        for i in range(self.depth - 1, -1, -1):
            self.backprop_memory(self.path[i], reward)
            reward = 1 - reward

    def backprop_memory(self, memory: Typing.StateDataDtype, reward: Typing.MCTSFloatDtype):
        # print(f"Memory:\n{memory}")
        # print(f"Memory dtype:\n{memory.dtype}")
        board = memory.board
        bestAction = memory.bestAction

        state_data = self.states[self.tobytes(board)][0]

        state_data.visits += 1                           # update n count
        state_data.rewards += reward                     # update state value
        if bestAction[0] == -1:
            return

        r, c = bestAction
        state_data.stateAction[..., r, c] += [1, reward]  # update state-action count / value

    def tobytes(self, arr: Typing.nbBoard):
        return ''.join(map(str, map(np.int8, np.nditer(arr))))




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
