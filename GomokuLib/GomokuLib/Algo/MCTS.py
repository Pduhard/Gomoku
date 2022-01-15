from __future__ import annotations
import copy
from os import stat
from time import perf_counter
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
        self.state_actions = {}    # Dict of List: [visit count, rewards sum, rewards / visitcount]
        self.c = c
        self.mcts_iter = iter
        
        # self.state_actions_shape = (3, 19, 19)
        # self.states = {}            # Dict of List: [visit count, rewards sum, rewards / visitcount]
        # self.state_actions = {}     # Visit count of pairs state/action

        self.engine = GomokuGUI(None, 19)

    def __call__(self, engine: Gomoku, state: np.ndarray, actions: np.ndarray) -> np.ndarray:

        print("\n[MCTS Object]\n")
        t = perf_counter()
        for i in range(5):

            self.engine.update(engine)
            self.mcts(state, actions)
            print(i)
        sa = self.state_actions[state.tobytes()]
        print('tt :', (perf_counter() - t) * 1000)
        print(sa[0])
        print(sa[1])
        # exit(0)
        return np.argmax(sa[0][1] / sa[0][0])

    def mcts(self, state: np.ndarray, actions: np.ndarray, mcts_iter: int = 0):

        print(f"\n[MCTS function {mcts_iter}]\n")

        path = []
        
        statehash = state.tobytes()
        while statehash in self.state_actions:
            
            bestaction = self.selection(statehash, actions)
            print(f"selection {bestaction.shape} {bestaction}")
            
            bestaction = (
                bestaction // self.engine.board_size[1],
                bestaction % self.engine.board_size[1]
            )
            self.engine.apply_action(GomokuAction(bestaction[0], bestaction[1]))
            self.engine.next_turn()
            self.engine.drawUI()
            
            path.append((statehash, bestaction))
            if self.engine.isnotover() is False:
                print('its over in mcts')
                exit(0)
            state = self.engine.state.board
            statehash = state.tobytes()

        brow, bcol = actions.shape
        self.state_actions[statehash] = [np.zeros((2, brow, bcol)), 1]
        # print(f"self.state_actions[statehash]: {self.state_actions[statehash]}")

        reward = self.evaluate(state)

        for p in path:
            # print(p)
            statehash, bestaction = p
            r, c = bestaction
            sa = self.state_actions[statehash]
            sa[0][..., r, c] += [1, reward]
            sa[1] += 1
        return


    def selection(self, statehash: str, actions: np.ndarray):
        """
            wi/ni + c * sqrt( ln(N) / ni )
        """
        (saVisits, saValue), sVisits  = self.state_actions[statehash]
        ucbs = saValue + self.c * np.sqrt(np.log10(sVisits) / (saVisits + 1))

        # print(ucbs)
        return np.argmax(ucbs * actions)

    # def heuristic(self, state):
    #     pass

    def evaluate(self, state):
        rd = np.random.random()
        print(rd)
        return rd
