from __future__ import annotations
import copy
from os import stat
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Union


import numpy as np

from GomokuLib.Player import Human

from GomokuLib.Game.Action import GomokuAction


from .AbstractAlgorithm import AbstractAlgorithm
from ..Game.GameEngine import GomokuGUI
from ..Game.GameEngine import Gomoku

class   MCTS(AbstractAlgorithm):

    engine: Gomoku = None

    def __init__(self, c: float = np.sqrt(2), iter: int = 1000) -> None:
        super().__init__()
        """
            State visite
            action / state visite
            policy
            heuristic (v ou random)
        """
        # self.states = {}           # Visit count of states
        self.states = {} # Dict of List: [
            # s visit count,
            # s rewards sum,
            # np.array(([
            #   sa visit count,
            #   sa rewards sum
            # ], 19, 19)],
            # np.array(([
            #   sa amaf count,
            #   sa amaf sum
            # ], 19, 19)]
        self.c = c
        self.mcts_iter = iter
        
        # self.states_shape = (3, 19, 19)
        # self.states = {}            # Dict of List: [visit count, rewards sum, rewards / visitcount]
        # self.states = {}     # Visit count of pairs state/action

        self.engine = GomokuGUI(None, 19)

    def __call__(self, engine: Gomoku, state: np.ndarray, actions: np.ndarray) -> np.ndarray:

        print("\n[MCTS Object]\n")
        t = perf_counter()
        for i in range(20000):

            self.engine.update(engine)
            self.mcts(state, actions, i)

        sa = self.states[state.tobytes()][2]
        print('tt :', (perf_counter() - t) * 1000)
        print(((sa[0])).astype(np.uint32))
        # print(sa[1])
        # exit(0)
        return np.argmax(sa[1] / sa[0])

    def mcts(self, state: np.ndarray, actions: np.ndarray, mcts_iter: int = 0):

        print(f"\n[MCTS function {mcts_iter}]\n")

        path = []

        statehash = state.tobytes()
        # print(f"statehash: {statehash.hex()}")
        while statehash in self.states and not self.engine.isover():

            actions = self.engine.get_actions()

            bestaction = self.selection(statehash, actions, mcts_iter)
            # print(f"selection {bestaction.shape} {bestaction}")
            self.engine.apply_action(GomokuAction(bestaction[0], bestaction[1]))
            self.engine.next_turn()
            self.engine.drawUI()

            path.append((statehash, bestaction))
            if self.engine.isover():
                print(self.engine.state.board)
                print('its over in mcts')
                sleep(10)
                exit(0)

            state = self.engine.state.board
            statehash = state.tobytes()
            mcts_iter += 1


        if self.engine.isover():
            new_state_actions = None
            new_amaf = None
            reward = 1
        else:
            brow, bcol = actions.shape
            new_state_actions = np.ones((2, brow, bcol))
            new_amaf = np.ones((2, brow, bcol))
            reward = self.evaluate(state)
            amaf_masks = [np.zeros_like(new_amaf), np.zeros_like(new_amaf)]

        self.states[statehash] = [1, reward, new_state_actions, new_amaf]
        # print(f"self.states[statehash]: {self.states[statehash]}")
        # print("leaf node evaluation end")
        
        amaf_idx = 0
        for p in path[::-1]:
            # print(f"path: {p}")
            statehash, bestaction = p
            r, c = bestaction
            state_data = self.states[statehash]
            amaf_masks[amaf_idx][..., r, c] += [1, reward]

            state_data[0] += 1                       # update visit count
            state_data[1] += reward                  # update state value
            state_data[2][..., r, c] += [1, reward]  # update state-action count / value
            state_data[3] += amaf_masks[amaf_idx]    # update amaf count / value

            amaf_idx ^= 1
            reward *= -1

        return


    def selection(self, statehash: str, actions: np.ndarray, mcts_iter: int):
        """
            wi/ni + c * sqrt( ln(N) / ni )
        """
        s_visits, s_value, (sa_visits, sa_value), (amaf_visits, amaf_value)  = self.states[statehash]

        exp_rate = self.c * np.sqrt(np.log(s_visits) / sa_visits)
        amaf = amaf_value / amaf_visits
        sa = sa_value / sa_visits
        beta = np.sqrt(1 / (1 + 3 * mcts_iter))
        quality = beta * amaf + (1 - beta) * sa

        ucbs = quality + exp_rate
        ucbs *= actions
        # return np.random.choice(np.argwhere(ucbs == np.amax(ucbs)))
        bestactions = np.argwhere(ucbs == np.amax(ucbs))
        return bestactions[np.random.randint(len(bestactions))]

    # def heuristic(self, state):
    #     pass

    def evaluate(self, state):
        rd = np.random.random()
        # print(rd)
        return rd
