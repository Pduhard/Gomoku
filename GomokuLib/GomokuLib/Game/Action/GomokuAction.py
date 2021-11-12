from GomokuLib.Game.Action.AbstractAction import AbstractAction

class   GomokuAction(AbstractAction):

    action = None

    def __init__(self, row, col) -> None:
        self.action = (row, col)