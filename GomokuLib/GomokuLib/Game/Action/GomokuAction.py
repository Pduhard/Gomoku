from __future__ import annotations
from typing import TYPE_CHECKING

from .AbstractAction import AbstractAction

class GomokuAction(AbstractAction):

    action: tuple = None

    def __init__(self, row: int, col: int) -> None:
        self.action = (row, col)

    # def __hash__(self) -> tuple:
    #     print(self.action)
    #     return self.action
