from .MCTSLazy import MCTSLazy
from .MCTSAMAF import MCTSAMAF


class MCTSAMAFLazy(MCTSLazy, MCTSAMAF):

    def __init__(self, *args, **kwargs) -> None:
        super(MCTSLazy, self).__init__(*args, **kwargs)
