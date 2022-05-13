import numpy as np
from numba import njit

from .MCTS import MCTS

# @njit(vectorize=True)
# def get_amaf_quality():


class MCTSAMAF(MCTS):

    def __str__(self):
        return f"MCTSAMAF with: Action-Move As First ({self.mcts_iter} iter)"

    def get_quality(self, state_data: list, mcts_iter: int, **kwargs) -> np.ndarray:
        """
            quality(s, a) = rewards(s, a)     / (visits(s, a)     + 1)
            AMAF(s, a)    = rewardsAMAF(s, a) / (visitsAMAF(s, a) + 1)
            0 < beta < 1

            AMAFQuality(s, a) = beta * AMAF(s, a) + (1 - beta) * quality(s, a)
        """
        sa_n, sa_v = state_data['StateAction']
        amaf_n, amaf_v = state_data['AMAF']

        sa = sa_v / (sa_n + 1)
        amaf = amaf_v / (amaf_n + 1)
        beta = np.sqrt(1 / (1 + 3 * mcts_iter))
        return beta * amaf + (1 - beta) * sa

    def expand(self):
        memory = super().expand()
        memory.update({
            'AMAF': np.zeros((2, self.brow, self.bcol))
        })
        return memory

    def backpropagation(self, path: list, reward: float):

        self.amaf_masks = np.zeros((2, 2, self.brow, self.bcol))    # sAMAF_v and sAMAF_n for 2 players
        super().backpropagation(path, reward)

    def backprop_memory(self, memory: tuple, reward: float):
        player_idx, statehash, bestaction = memory

        state_data = self.states[statehash]

        state_data['Visits'] += 1  # update n count
        state_data['Rewards'] += reward  # update state value
        if bestaction is None:
            return

        r, c = bestaction.action
        state_data['StateAction'][..., r, c] += [1, reward]  # update state-action count / value

        self.amaf_masks[player_idx, ..., r, c] += [1, reward]
        state_data['AMAF'] += self.amaf_masks[player_idx]    # update amaf count / value
