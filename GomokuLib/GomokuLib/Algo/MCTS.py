from __future__ import annotations
import copy
from os import stat
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Union


import numpy as np

from GomokuLib.Player import Human

from GomokuLib.Game.Action import GomokuAction

from GomokuLib.Game.State import GomokuState


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

        self.engine = Gomoku(None, 19)
        # self.engine = GomokuGUI(None, 19)

    def __call__(self, engine: Gomoku, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        print("\n[MCTS Object]\n")

        self.bs = np.array(self.engine.board_size)
        # self.bsr, self.bsc = self.bs

        t = perf_counter()
        for i in range(2000):
            # print(i, " ", self.engine.rules_fn)
            self.engine.update(engine)
            # print(i, " ", self.engine.rules_fn)
            self.mcts(state, actions, i)

        sa = self.states[state.tobytes()][2]
        print('tt :', (perf_counter() - t) * 1000)
        print(((sa[0])).astype(np.uint32))
        
        print("MCTS state:", state)
        print(sa[0])
        print(sa[1])
        # exit(0)
        arg = np.nan_to_num(sa[1] / sa[0])
        # print("action take: ", arg // 19, arg % 19)
        return arg

    def mcts(self, state: np.ndarray, actions: np.ndarray, mcts_iter: int = 0):

        print(f"\n[MCTS function {mcts_iter}]\n")

        path = []

        statehash = state.tobytes()
        # print(f"statehash: {statehash.hex()}")
        while statehash in self.states and not self.engine.isover():

            actions = self.engine.get_actions()
            # print("actions", actions.shape, actions)

            bestaction = self.selection(statehash, actions, mcts_iter)
            # print(f"selection {bestaction.shape} {bestaction}")
            self.engine.apply_action(GomokuAction(bestaction[0], bestaction[1]))
            self.engine.next_turn()
            # self.engine.drawUI()

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
            reward = [1]
        else:
            brow, bcol = actions.shape
            new_state_actions = np.zeros((2, brow, bcol))
            new_amaf = np.zeros((2, brow, bcol))
            rewards = self.evaluate_random_rollingout(state)
            amaf_masks = [np.zeros_like(new_amaf), np.zeros_like(new_amaf)]

        self.states[statehash] = [1, rewards[0], new_state_actions, new_amaf]
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

            state_data[0] += 1                       # update visit count
            state_data[1] += reward                  # update state value
            state_data[2][..., r, c] += [1, reward]  # update state-action count / value
            state_data[3] += amaf_masks[player_idx]    # update amaf count / value

            player_idx ^= 1

        return


    def selection(self, statehash: str, actions: np.ndarray, mcts_iter: int):
        """
            wi/ni + c * sqrt( ln(N) / ni )
        """
        s_visits, s_value, (sa_visits, sa_value), (amaf_visits, amaf_value)  = self.states[statehash]

        exp_rate = np.nan_to_num(self.c * np.sqrt(np.log(s_visits) / sa_visits))
        amaf = np.nan_to_num(amaf_value / amaf_visits)
        sa = np.nan_to_num(sa_value / sa_visits)
        beta = np.sqrt(1 / (1 + 3 * mcts_iter))
        quality = beta * amaf + (1 - beta) * sa

        ucbs = quality + exp_rate
        ucbs *= actions
        # return np.random.choice(np.argwhere(ucbs == np.amax(ucbs)))
        bestactions = np.argwhere(ucbs == np.amax(ucbs))
        # print(bestactions)
        return bestactions[np.random.randint(len(bestactions))]

    # def heuristic(self, state):
    #     pass

    def evaluate_random_rollingout(self, board: np.ndarray):

        # return [1, 1]
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
