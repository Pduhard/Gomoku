from __future__ import annotations
import copy
import random
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

    def get_actions(self) -> np.ndarray:
        return self.engine.get_actions()
    
    def _lazy_get_actions(self) -> np.ndarray:
        return np.ones(self.engine.board_size, dtype=bool)

    def mcts(self, mcts_iter: int):

        print(f"\n[MCTS function {mcts_iter}]\n")

        path = []
        mcts_idx = self.engine.player_idx
        state = self.engine.state.board
        statehash = state.tobytes()
        # print(f"statehash: {statehash.hex()}")
        while statehash in self.states and not self.engine.isover():

            state_data = self.states[statehash]
            # print("actions: ", state_data[4])

            ucbs = self.get_policy(state_data, mcts_iter)
            bestaction = self.selection(ucbs, state_data)
            bestGAction = GomokuAction(bestaction[0], bestaction[1])

            print(f"selection {bestaction}")
            if not self.engine.is_valid_action(bestGAction):
                print(f"Not a fucking valid action in mcts: {bestaction}")
                raise Exception

            path.append((self.engine.player_idx, statehash, bestaction))
            self.engine.apply_action(bestGAction)
            self.engine.next_turn()

            if self.engine.isover():
                print(self.engine.state.board)
                print('its over in mcts')
                exit(0)

            state = self.engine.state.board
            statehash = state.tobytes()
            mcts_iter += 1

        end_game = self.engine.isover()
        # self.expand()
        # reward = self.reward(end_game)
        path.append((self.engine.player_idx, statehash, None))

        if end_game:
            new_state_actions = None
            new_amaf = None
            if self.engine.winner >= 0:
                rewards = [self.engine.winner == mcts_idx, self.engine.winner != mcts_idx]
            else:
                rewards = [0.5, 0.5]

        else:
            brow, bcol = self.engine.board_size
            new_state_actions = np.zeros((2, brow, bcol))
            new_amaf = np.zeros((2, brow, bcol))
            rewards = self.evaluate_random_rollingout(state)

        self.states[statehash] = [1, rewards[0], new_state_actions, new_amaf, self.get_actions()]
        # print(f"self.states[statehash]: {self.states[statehash]}")
        # print("leaf node evaluation end")

        self.backpropagation(path, rewards)
        return

    # def new_memory(self, statehash, bestaction):
    #     return (self.engine.player_idx, statehash, bestaction)

    def backpropagation(self, memory: tuple, rewards: list):

        amaf_masks = np.zeros((2, 2, self.engine.board_size[0], self.engine.board_size[1]))
        for player_idx, statehash, bestaction in memory[::-1]:

            reward = rewards[player_idx]
            state_data = self.states[statehash]

            state_data[0] += 1                       # update n count
            state_data[1] += reward                  # update state value
            if bestaction is None:
                continue

            r, c = bestaction
            state_data[2][..., r, c] += [1, reward]  # update state-action count / value
            amaf_masks[player_idx, ..., r, c] += [1, reward]
            state_data[3] += amaf_masks[player_idx]    # update amaf count / value


    def get_policy(self, state_data: list, mcts_iter: int) -> np.np.ndarray:
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

    def selection(self, policy: np.ndarray, *args) -> tuple:

        bestactions = np.argwhere(policy == np.amax(policy))
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
        actions = np.meshgrid(np.arange(self.engine.board_size[0]), np.arange(self.engine.board_size[1]))
        actions = np.array(actions).T.reshape(
            self.engine.board_size[0] * self.engine.board_size[1], 2)

        mcts_idx = self.engine.player_idx
        while not self.engine.isover():

            np.random.shuffle(actions)
            i = 0
            while not self.engine.is_valid_action(GomokuAction(*actions[i])):
                i += 1

            self.engine.apply_action(GomokuAction(*actions[i]))
            self.engine.next_turn()

        winner = self.engine.winner
        if winner == -1:
            return [0.5, 0.5]
        else:
            return [self.engine.winner == mcts_idx, self.engine.winner != mcts_idx]

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
