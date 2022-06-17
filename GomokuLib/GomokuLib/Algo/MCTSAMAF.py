import numpy as np
from numba import njit
import numba as nb

from .MCTS import MCTS


# @njit(vectorize=True)
@nb.vectorize('float64(float64, float64, float64, float64, int8)')
def get_amaf_quality(sa_n: np.ndarray, sa_v: np.ndarray, amaf_n: np.ndarray, amaf_v: np.ndarray, mcts_iter: int):
    sa = sa_v / (sa_n + 1)
    amaf = amaf_v / (amaf_n + 1)
    beta = np.sqrt(1 / (1 + 3 * mcts_iter))
    return beta * amaf + (1 - beta) * sa


class MCTSAMAF(MCTS):

    def __str__(self):
        return f"MCTSAMAF with: Action-Move As First ({self.mcts_iter} iter)"

    def get_quality(self, state_data: list, mcts_iter: int, **kwargs) -> np.ndarray:
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
        return get_amaf_quality(*state_data['stateAction'], *state_data['AMAF'], mcts_iter)

        # sa_n, sa_v = state_data['stateAction']
        # amaf_n, amaf_v = state_data['AMAF']
        # sa = sa_v / (sa_n + 1)
        # amaf = amaf_v / (amaf_n + 1)
        # beta = np.sqrt(1 / (1 + 3 * mcts_iter))
        # return beta * amaf + (1 - beta) * sa

    def expand(self):
        memory = super().expand()
        memory.update({
            'AMAF': np.zeros((2, self.brow, self.bcol))
        })
        return memory

    def backpropagation(self, path: list):

        self.amaf_masks = np.zeros((2, 2, self.brow, self.bcol))    # sAMAF_v and sAMAF_n for 2 players
        super().backpropagation(path)

    def backprop_memory(self, memory: tuple, reward: float):
        player_idx, statehash, bestaction = memory

        state_data = self.states[statehash]

        state_data['visits'] += 1  # update n count
        state_data['rewards'] += reward  # update state value
        if bestaction is None:
            return

        r, c = bestaction.action
        state_data['stateAction'][..., r, c] += [1, reward]  # update state-action count / value

        self.amaf_masks[player_idx, ..., r, c] += [1, reward]
        state_data['AMAF'] += self.amaf_masks[player_idx]    # update amaf count / value
