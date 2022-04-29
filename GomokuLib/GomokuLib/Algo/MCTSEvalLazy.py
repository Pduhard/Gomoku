from .MCTSLazy import MCTSLazy
from .MCTSEval import MCTSEval


class MCTSEvalLazy(MCTSLazy, MCTSEval):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"MCTSEvalLazy with: Pruning / Heuristics | Progressive/Lazy valid action checking ({self.mcts_iter} iter)"
