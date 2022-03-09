from __future__ import annotations
import copy
from os import stat
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Dict, Tuple, Union


import numpy as np
from numba import njit

from GomokuLib.Player import Human

from GomokuLib.Game.Action import GomokuAction

from GomokuLib.Game.State import GomokuState


from .AbstractAlgorithm import AbstractAlgorithm
from ..Game.GameEngine import GomokuGUI
from ..Game.GameEngine import Gomoku

# @njit(fastmath=True)
# def njit_selection(s_n, sa_n, sa_v, amaf_n, amaf_v, c, mcts_iter, actions):

#     exp_rate = c * np.sqrt(np.log(s_n) / (sa_n + 1))
#     amaf = amaf_v / (amaf_n + 1)
#     sa = sa_v / (sa_n + 1)
#     beta = np.sqrt(1 / (1 + 3 * mcts_iter))
#     quality = beta * amaf + (1 - beta) * sa

#     ucbs = quality + exp_rate
#     ucbs *= actions
#     # return np.random.choice(np.argwhere(ucbs == np.amax(ucbs)))
#     bestactions = np.argwhere(ucbs == np.amax(ucbs))
#     # print(bestactions)
#     return bestactions[np.random.randint(len(bestactions))]

class   MCTS(AbstractAlgorithm):

    engine: Gomoku = None

    def __init__(self, c: float = np.sqrt(2), iter: int = 1000, lazy: bool = True) -> None:
        super().__init__()
        """
            State ne
            action / state ne
            policy
            heuristic (v ou random)

            self.states : 
                Dict of List: 
                    State visit
                    State reward
                    State/actions visit/reward for each cells (2*19*19)
                    State/actions amaf visit/reward for each cells (2*19*19)
                    Actions (1*19*19)
        """
        # self.states = {}           # n count of states
        self.states: dict = {}
        self.c = c
        self.mcts_iter = iter
        if lazy:
            self.get_actions = self._lazy_get_actions
            self.selection = self._lazy_selection
        else:
            self.get_actions = self._get_actions
            self.selection = self._selection
        # self.states_shape = (3, 19, 19)
        # self.states = {}            # Dict of List: [n count, rewards sum, rewards / ncount]
        # self.states = {}     # n count of pairs state/action

        self.engine = Gomoku(None, 19)
        # self.engine = GomokuGUI(None, 19)

    def __call__(self, game_engine: Gomoku) -> np.ndarray:
        print("\n[MCTS Object __call__()]\n")

        self.bs = np.array(self.engine.board_size)
        # self.bsr, self.bsc = self.bs

        for i in range(self.mcts_iter):
            self.engine.update(game_engine)
            self.mcts(i)

        statehash = game_engine.state.board.tobytes()
        sa = self.states[statehash][2]

        arg = np.nan_to_num(sa[1] / (sa[0] + 1))

        return arg

    def _get_actions(self) -> np.ndarray:
        return self.engine.get_actions()
    
    def _lazy_get_actions(self) -> np.ndarray:
        return np.ones(self.engine.board_size, dtype=bool)

    def mcts(self, mcts_iter: int):

        print(f"\n[MCTS function {mcts_iter}]\n")

        path = []

        state = self.engine.state.board
        statehash = state.tobytes()
        # print(f"statehash: {statehash.hex()}")
        while statehash in self.states and not self.engine.isover():

            state_data = self.states[statehash]
            # print("actions: ", state_data[4])

            ucbs = self._compute_ucbs(state_data, mcts_iter)
            bestaction = self.selection(ucbs, state_data[4])
            bestGAction = GomokuAction(bestaction[0], bestaction[1])

            print(f"selection {bestaction}")
            if not self.engine.is_valid_action(bestGAction):
                print(f"Not a fucking valid action in mcts: {bestaction}")
                raise Exception

            self.engine.apply_action(bestGAction)
            self.engine.next_turn()

            path.append((statehash, bestaction))
            if self.engine.isover():
                print(self.engine.state.board)
                print('its over in mcts')
                exit(0)

            state = self.engine.state.board
            statehash = state.tobytes()
            mcts_iter += 1


        if self.engine.isover():
            new_state_actions = None
            new_amaf = None
            reward = [1]
        else:
            brow, bcol = self.engine.board_size
            new_state_actions = np.zeros((2, brow, bcol))
            new_amaf = np.zeros((2, brow, bcol))
            rewards = self.evaluate_random_rollingout(state)
            amaf_masks = [np.zeros_like(new_amaf), np.zeros_like(new_amaf)]

        self.states[statehash] = [1, rewards[0], new_state_actions, new_amaf, self.get_actions()]
        # print(f"self.states[statehash]: {self.states[statehash]}")
        # print("leaf node evaluation end")

        player_idx = 0
        for p in path[::-1]:
            # print(f"path: {p}")
            reward = rewards[player_idx]
            statehash, bestaction = p
            r, c = bestaction
            state_data = self.states[statehash]
            
            amaf_masks[player_idx][..., r, c] += [1, reward]

            state_data[0] += 1                       # update n count
            state_data[1] += reward                  # update state value
            state_data[2][..., r, c] += [1, reward]  # update state-action count / value
            state_data[3] += amaf_masks[player_idx]    # update amaf count / value

            player_idx ^= 1

        return

    def _compute_ucbs(self, state_data: list, mcts_iter: int) -> np.np.ndarray:
        """
            wi/ni + c * sqrt( ln(N) / ni )
        """
        s_n, _, (sa_n, sa_v), (amaf_n, amaf_v), actions = state_data
        # return njit_selection(s_n, sa_n, sa_v, amaf_n, amaf_v, self.c, mcts_iter, actions)

        exp_rate = self.c * np.sqrt(np.log(s_n) / (sa_n + 1))
        amaf = amaf_v / (amaf_n + 1)
        sa = sa_v / (sa_n + 1)
        beta = np.sqrt(1 / (1 + 3 * mcts_iter))
        quality = beta * amaf + (1 - beta) * sa

        ucbs = quality + exp_rate
        return ucbs * actions

    def _lazy_selection(self, ucbs: np.ndarray, actions: np.ndarray) -> tuple:

        rows, cols = np.unravel_index(np.argsort(ucbs, axis=None), ucbs.shape)
        for x, y in zip(rows[::-1], cols[::-1]):
            print("ucbs", ucbs[x, y], np.amax(ucbs))
            if self.engine.is_valid_action(GomokuAction(x, y)):
                return (x, y)
            actions[x, y] = 0
        return None

    def _selection(self, ucbs: np.ndarray, *args) -> tuple:

        bestactions = np.argwhere(ucbs == np.amax(ucbs))
        return bestactions[np.random.randint(len(bestactions))]

    """
    
    exp_rate = c * np.sqrt(np.log(s_n) / (sa_n + 1))
    amaf = amaf_v / (amaf_n + 1)
    sa = sa_v / (sa_n + 1)
    beta = np.sqrt(1 / (1 + 3 * mcts_iter))
    quality = beta * amaf + (1 - beta) * sa

    ucbs = quality + exp_rate
    ucbs *= actions
    # return np.random.choice(np.argwhere(ucbs == np.amax(ucbs)))
    bestactions = np.argwhere(ucbs == np.amax(ucbs))
    # print(bestactions)
    return bestactions[np.random.randint(len(bestactions))]
    """

    def evaluate_random_rollingout(self, board: np.ndarray):

        return [1, 1]
        # print(self.engine.isover())
        i_roll = 0
        while not self.engine.isover():
            actions = self.engine.get_actions()
            actions = np.argwhere(actions)
            # print(actions)
            rd = np.random.randint(len(actions))
            ranr, ranc = actions[rd]
            action = GomokuAction(ranr, ranc)
            i = 0
            while not self.engine.is_valid_action(action):
                # if (i > 10000):
                #     print(self.engine.state.board)
                #     exit(0)
                i += 1
                # actions.pop(rd)
                rd = np.random.randint(len(actions))
                ranr, ranc = actions[rd]
                action = GomokuAction(ranr, ranc)
                
            # exit(0)
            # ranr, ranc = np.random.randint(19, size=2)
            # # print('oui', ranr, ranc, ranr)
            # action = GomokuAction(ranr, ranc)
            # i = 0
            # while self.engine.is_valid_action(action):
            #     ranr, ranc = np.random.randint(19, size=2)
            #     action = GomokuAction(ranr, ranc)
            #     i += 1
            # print(i)
            self.engine.apply_action(action)
            self.engine.next_turn()
            i_roll += 1
            # print(i_roll)
            if i_roll == 5:
                # print("seeeeeeeeeeeeeeee")
                return self.evaluate(self.engine.state.board)
            # print(self.engine.state.board)



        winner = self.engine.winner
        if winner == -1:
            return [0.5, 0.5]
        else:
            return [1, 0] if self.engine.winner == 0 else [0, 1]

    def evaluate(self, board: np.ndarray):

        # board = np.ones_like(board)
        count = np.sum(board, axis=(1, 2))
        # print(count.shape, count)
        countscoef = np.nan_to_num(count / np.sum(count))

        # coords = board[board != 0]
        # coords = np.transpose(np.nonzero(board))
        coords = (
            np.argwhere(board[0]) - self.bs // 2,
            np.argwhere(board[1]) - self.bs // 2
        )
        # coord = board[board.astype(np.bool)]
        # print(board.shape, board)
        # print("coords: ", coords)
        # print("coord")

        dists = np.nan_to_num([
            np.mean(np.sum(np.abs(coords[0]), axis=-1)),
            np.mean(np.sum(np.abs(coords[1]), axis=-1))
        ])

        # dists1 = np.sum(abs(coords[1]), axis=0)
        # print("dists :", dists)

        distscoef = np.nan_to_num(dists / np.sum(dists))
        # print("distcoef: ", distscoef)
        # print("countcoef: ", countscoef)
        # exit(0)

        return (countscoef + distscoef) / 2
