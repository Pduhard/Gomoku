from __future__ import annotations
from typing import TYPE_CHECKING

from .AbstractAction import AbstractAction

class   GomokuAction(AbstractAction):

    action = None

    def __init__(self, row, col) -> None:
        self.action = (row, col)