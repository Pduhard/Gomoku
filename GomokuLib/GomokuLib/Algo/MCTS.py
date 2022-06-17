import numpy as np

from GomokuLib import Typing
from GomokuLib.Game.GameEngine import Gomoku

class MCTS:

    def __init__(self,
                 engine: Gomoku,
                 iter: int = 0,
                 *args, **kwargs
                 ) -> None:
        """
            self.states : 
                Dict of List: 
                    State visit
                    State reward
                    State/actions: visit/reward for each cells (2*19*19)
                    actions (1*19*19)

        """

        self.states: dict = {}
        if not engine:
            raise Exception("[MCTS error] No engine passed")
        self.engine = engine.clone()
        self.c = np.sqrt(2)
        self.mcts_iter = iter if iter else 5000

        self.board_size = self.engine.board_size
        self.brow, self.bcol = self.engine.board_size
        self.cells_count = self.brow * self.bcol

    def __str__(self):
        return f"Classic MCTS ({self.mcts_iter} iter)"

    def init(self):
        self.reset()

    def reset(self):
        self.states = {}

    def __call__(self, game_engine: Gomoku) -> tuple:
        """
            Reward in range [0, 1]
            Policy in range [0, 1)
        """
        print(f"\n[MCTS Object __call__()] -> {self.mcts_iter}\n")

        self.max_depth = 0
        for i in range(self.mcts_iter):
            self.engine.update(game_engine)

            self.mcts(i)

            if self.depth + 1 > self.max_depth:
                self.max_depth = self.depth + 1

        state_data = self.states[game_engine.board.tobytes()]
        sa_n, sa_v = state_data['stateAction']

        self.mcts_policy = sa_v / (sa_n + 1)

        self.engine.update(game_engine)
        self.gAction = self.selection(self.mcts_policy, state_data)

        return self.mcts_policy, self.gAction

    def get_state_data(self, game_engine):
        statehash = game_engine.board.tobytes()
        if statehash in self.states:
            state_data = self.states[game_engine.board.tobytes()]
            state_data['max_depth'] = self.max_depth
            return {
                'mcts_state_data': [state_data]
            }
        else:
            return {}

    def mcts(self, mcts_iter: int):

        self.depth = 0
        path = []
        self.current_board = self.engine.board
        statehash = self.current_board.tobytes()
        self.bestGAction = None
        self.end_game = self.engine.isover()
        while statehash in self.states and not self.end_game:

            state_data = self.states[statehash]

            policy = self.get_policy(state_data, mcts_iter=mcts_iter)
            self.bestGAction = self.selection(policy, state_data)

            path.append(self.new_memory(statehash))
            self.engine.apply_action(self.bestGAction)
            self.engine.next_turn()
            self.depth += 1

            self.current_board = self.engine.board
            statehash = self.current_board.tobytes()

            self.end_game = self.engine.isover()

        self.draw = self.engine.winner == -1

        self.bestGAction = None
        path.append(self.new_memory(statehash))

        self.states[statehash] = self.expand()

        self.backpropagation(path)
        return

    def get_actions(self) -> np.ndarray:
        return self.engine.get_actions()

    def get_quality(self, state_data: list, **kwargs) -> np.ndarray:
        """
            quality(s, a) = reward(s, a) / (visits(s, a) + 1)
        """
        sa_n, sa_v = state_data['stateAction']
        return sa_v / (sa_n + 1)

    def get_exp_rate(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) = c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )
        """
        return self.c * np.sqrt(np.log(state_data['visits']) / (state_data['stateAction'][0] + 1))

    def get_policy(self, state_data: list, **kwargs) -> np.ndarray:
        """
            ucb(s, a) = (quality(s, a) + exp_rate(s, a)) * valid_action(s, a)
        """
        return self.get_quality(state_data, **kwargs) + self.get_exp_rate(state_data, **kwargs)

    def selection(self, policy: np.ndarray, state_data, *args) -> tuple:

        policy *= state_data['actions']     # Avaible actions
        bestactions = np.argwhere(policy == np.amax(policy))
        bestaction = bestactions[np.random.randint(len(bestactions))]
        return np.array(bestaction, dtype=Typing.TupleDtype)

    def expand(self):
        actions = self.get_actions()
        self.reward = self.award_end_game() if self.end_game else self.award()
        return {
            'visits': 1,
            'rewards': 0,
            'stateAction': np.zeros((2, self.brow, self.bcol)),
            'actions': actions,
            'heuristic': self.reward
        }

    def new_memory(self, statehash: str):
        return self.engine.player_idx, statehash, self.bestGAction

    def backpropagation(self, path: list):

        reward = self.reward
        for mem in path[::-1]:
            self.backprop_memory(mem, reward)
            reward = 1 - reward

    def backprop_memory(self, memory: tuple, reward: float):
        _, statehash, bestaction = memory

        state_data = self.states[statehash]

        state_data['visits'] += 1                           # update n count
        state_data['rewards'] += reward                     # update state value
        if bestaction is None:
            return

        r, c = bestaction
        state_data['stateAction'][..., r, c] += [1, reward]  # update state-action count / value

    def award(self):
        return 0.5

    def award_end_game(self):
        if self.draw:
            return 0.5
        return 1 if self.engine.winner == self.engine.player_idx else 0
