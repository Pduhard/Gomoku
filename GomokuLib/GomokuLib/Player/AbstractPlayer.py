from abc import ABCMeta, abstractmethod
from typing import Any

import GomokuLib
from GomokuLib import AbstractGameEngine
# from GomokuLib.Game.Action.AbstractAction import AbstractAction
# from GomokuLib.Game.State.AbstractState import AbstractState

# import GomokuLib.Game.GameEngine.AbstractGameEngine as AbstractGameEngine

# try:
#     from GomokuLib.Game.GameEngine.AbstractGameEngine import AbstractGameEngine
# except:
#     AbstractGameEngine = Any
#     print('Hatt')

# a = AbstractGameEngine()
# print(a)


class AbstractPlayer(metaclass=ABCMeta):

    engine: AbstractGameEngine = None

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def play_turn(self, state: GomokuLib.AbstractState,
                  actions: list[GomokuLib.AbstractAction]) -> GomokuLib.AbstractAction:
        pass

    def init_engine(self, engine: GomokuLib.AbstractGameEngine) -> None:
        self.engine = engine
