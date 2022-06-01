from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from .MCTSLazy import MCTSLazy
from .MCTSEval import MCTSEval


class MCTSEvalLazy(MCTSLazy, MCTSEval):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"MCTSEvalLazy({self.mcts_iter} iter) with: Pruning / Heuristics | Progressive/Lazy valid action checking"

    def __call__(self, game_engine: Gomoku) -> tuple:
        print(f"\n[MCTSEvalLazy __call__() for {self.mcts_iter} iter]\n")
        return super().__call__(game_engine)
