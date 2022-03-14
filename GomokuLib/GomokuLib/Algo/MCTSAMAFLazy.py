from .MCTSLazy import MCTSLazy
from .MCTSAMAF import MCTSAMAF


class MCTSAMAFLazy(MCTSLazy, MCTSAMAF):

    def __init__(self) -> None:
        super(MCTSLazy, self).__init__()
