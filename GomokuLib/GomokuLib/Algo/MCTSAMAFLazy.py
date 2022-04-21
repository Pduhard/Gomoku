from .MCTSLazy import MCTSLazy
from .MCTSAMAF import MCTSAMAF


class MCTSAMAFLazy(MCTSLazy, MCTSAMAF):

    def __init__(self, *args, **kwargs) -> None:
        super(MCTSLazy, self).__init__(*args, **kwargs)

    def __str__(self):
        return f"MCTS with: Action-Move As First | progressive/Lazy valid action checking ({self.mcts_iter} iter)"
