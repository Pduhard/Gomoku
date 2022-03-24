import numpy as np
from .MCTS import MCTS


class MCTSAMAF(MCTS):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_quality(self, state_data: list, mcts_iter: int, **kwargs) -> np.ndarray:
        """
            quality(s, a) = rewards(s, a)     / (visits(s, a)     + 1)
            AMAF(s, a)    = rewardsAMAF(s, a) / (visitsAMAF(s, a) + 1)
            0 < beta < 1

            AMAFQuality(s, a) = beta * AMAF(s, a) + (1 - beta) * quality(s, a)
        """
        _, _, (sa_n, sa_v), _, (amaf_n, amaf_v) = state_data[:5]

        sa = sa_v / (sa_n + 1)
        amaf = amaf_v / (amaf_n + 1)
        beta = np.sqrt(1 / (1 + 3 * mcts_iter))
        return beta * amaf + (1 - beta) * sa

    def expand(self):
        memory = super().expand()

        amaf_values = None if self.end_game else np.zeros((2, self.brow, self.bcol))
        memory.append(amaf_values)
        return memory

    def backpropagation(self, path: list, rewards: list):

        self.amaf_masks = np.zeros((2, 2, self.brow, self.bcol))
        super().backpropagation(path, rewards)

    def backprop_memory(self, memory: tuple, rewards: list):
        player_idx, statehash, bestaction = memory

        reward = rewards[player_idx]
        state_data = self.states[statehash]

        state_data[0] += 1  # update n count
        state_data[1] += reward  # update state value
        if bestaction is None:
            return

        # breakpoint()
        r, c = bestaction
        state_data[2][..., r, c] += [1, reward]  # update state-action count / value
        self.amaf_masks[player_idx, ..., r, c] += [1, reward]
        state_data[4] += self.amaf_masks[player_idx]    # update amaf count / value
