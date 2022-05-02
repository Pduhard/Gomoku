from __future__ import annotations
import copy
import random
from os import stat
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Dict, Tuple, Union


import numpy as np
# from numba import njit

from GomokuLib.Player import Human

from GomokuLib.Game.Action import GomokuAction

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

class MCTS(AbstractAlgorithm):

    def __init__(self,
                 engine: Gomoku,
                 c: float = np.sqrt(2),
                 iter: int = 1000,
                 *args, **kwargs
                 ) -> None:
        """
            self.states : 
                Dict of List: 
                    State visit
                    State reward
                    State/actions: visit/reward for each cells (2*19*19)
                    Actions (1*19*19)

        """
        super().__init__()

        self.states: dict = {}
        if not engine:
            raise Exception("[MCTS error] No engine passed")
        self.engine = engine.clone()
        self.c = c
        self.mcts_iter = iter

        self.board_size = self.engine.board_size
        self.brow, self.bcol = self.engine.board_size
        self.cells_count = self.brow * self.bcol

    def __str__(self):
        return f"Classic MCTS ({self.mcts_iter} iter)"

    def __call__(self, game_engine: Gomoku) -> tuple:
        """
            Reward in range [0, 1]
            Policy in range [0, 1)
        """
        print(f"\n[MCTS Object __call__()] -> {self.mcts_iter}\n")

        for i in range(self.mcts_iter):
            self.engine.update(game_engine)
            self.mcts(i)

        state_data = self.states[game_engine.state.board.tobytes()]
        sa_n, sa_v = state_data['StateAction']
        # sa_n, sa_v = state_data[2]

        self.mcts_policy = sa_v / (sa_n + 1)
        # print("self.mcts_policy (rewards sum / visit count):\n", self.mcts_policy)

        # print(f"MCTS  self.mcts_policy:\n{self.mcts_policy}")
        # print(f"Model self.mcts_policy:\n{state_data[5][0]}")
        # print(f"Model value :\n{state_data[5][1]}")

        self.engine.update(game_engine)
        self.gAction = None
        while not (self.gAction and game_engine.is_valid_action(self.gAction)):
            self.gAction = self.selection(self.mcts_policy, state_data)
            print(f"Ultimate __call__() selection:\n{self.gAction.action}")
            # print(f"Ultimate __call__() selection:\n{self.gAction.action} with self.mcts_policy={self.mcts_policy[self.gAction.action]}")

        return self.mcts_policy, self.gAction

    def get_state_data(self, engine):
        return {
            'mcts_state_data': self.states[engine.state.board.tobytes()],
        }

    def mcts(self, mcts_iter: int):

        # print(f"\n[MCTS function {mcts_iter}]\n")

        path = []
        self.mcts_idx = self.engine.player_idx
        self.current_board = self.engine.state.board
        statehash = self.current_board.tobytes()
        self.bestGAction = None
        # print(f"statehash: {statehash.hex()}")
        while statehash in self.states and not self.engine.isover():

            state_data = self.states[statehash]

            policy = self.get_policy(state_data, mcts_iter=mcts_iter)
            self.bestGAction = self.selection(policy, state_data)

            # print(f"selection {self.bestGAction.action}")
            # if not self.engine.is_valid_action(self.bestGAction):
            #     print(f"Not a fucking valid action in mcts: {self.bestGAction.action}")
            #     breakpoint()
            #     raise Exception

            path.append(self.new_memory(statehash, self.bestGAction.action))
            self.engine.apply_action(self.bestGAction)
            self.engine.next_turn()

            self.current_board = self.engine.state.board
            statehash = self.current_board.tobytes()
            mcts_iter += 1

        self.end_game = self.engine.isover()
        self.win = self.mcts_idx == self.engine.winner
        self.draw = self.engine.winner == -1

        path.append(self.new_memory(statehash, None))

        self.states[statehash] = self.expand()
        rewards = self.evaluate()

        self.backpropagation(path, rewards)
        return

    def get_actions(self) -> np.ndarray:
        return self.engine.get_actions()


    def get_quality(self, state_data: list, **kwargs) -> np.ndarray:
        """
            quality(s, a) = rewards(s, a) / (visits(s, a) + 1)
        """
        sa_n, sa_v = state_data['StateAction']
        # _, _, (sa_n, sa_v) = state_data[:3]
        return sa_v / (sa_n + 1)

    def get_exp_rate(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) = c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )
        """
        # s_n, _, (sa_n, _) = state_data[:3]
        return self.c * np.sqrt(np.log(state_data['Visits']) / (state_data['StateAction'][0] + 1))

    def get_policy(self, state_data: list, **kwargs) -> np.ndarray:
        """
            ucb(s, a) = (quality(s, a) + exp_rate(s, a)) * valid_action(s, a)
        """
        ucbs = self.get_quality(state_data, **kwargs) + self.get_exp_rate(state_data, **kwargs)
        return ucbs

    def selection(self, policy: np.ndarray, state_data, *args) -> tuple:

        policy *= state_data['Actions']     # Avaible actions
        # policy *= state_data[3]     # Avaible actions
        bestactions = np.argwhere(policy == np.amax(policy))
        bestaction = bestactions[np.random.randint(len(bestactions))]
        return GomokuAction(*bestaction)

    def expand(self):
        return {
            'Visits': 1,
            'Rewards': 0,
            'StateAction': np.zeros((2, self.brow, self.bcol)),
            'Actions': self.get_actions()
        }

    def new_memory(self, statehash, bestaction):
        return 0 if self.engine.player_idx == self.mcts_idx else 1, statehash, bestaction

    def backpropagation(self, path: list, rewards: list):

        for mem in path[::-1]:
            self.backprop_memory(mem, rewards)

    def backprop_memory(self, memory: tuple, rewards: list):

        player_idx, statehash, bestaction = memory

        reward = rewards[player_idx]
        state_data = self.states[statehash]

        state_data['Visits'] += 1                           # update n count
        state_data['Rewards'] += reward                     # update state value
        if bestaction is None:
            return

        r, c = bestaction
        state_data['StateAction'][..., r, c] += [1, reward]  # update state-action count / value

    def reset(self):
        self.states = {}

    def evaluate(self):
        return self.award_end_game() if self.end_game else self.award()

    def award_end_game(self):
        if self.draw:
            return [0.5, 0.5]
        return [1 if self.win else 0, 1 if not self.win else 0]

    def award(self):
        return self._evaluate_random_rollingout()

    def _evaluate_random_rollingout(self):
        return [0.5, 0.5]
    #     actions = np.meshgrid(np.arange(self.brow), np.arange(self.bcol))
    #     actions = np.array(actions).T.reshape(self.cells_count, 2)
    #
    #     self.mcts_idx = self.engine.player_idx
    #     while not self.engine.isover():
    #
    #         np.random.shuffle(actions)
    #         i = 0
    #         while not self.engine.is_valid_action(GomokuAction(*actions[i])):
    #             i += 1
    #
    #         self.engine.apply_action(GomokuAction(*actions[i]))
    #         self.engine.next_turn()
    #
    #     self.end_game = self.engine.isover()
    #     self.win = self.mcts_idx == self.engine.winner
    #     self.draw = self.engine.winner == -1
    #     return self.award_end_game()
