import GomokuLib
from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Game.Action.GomokuAction import GomokuAction
from GomokuLib.Player.Human import Human

if __name__ == "__main__":

    p1 = Human(verbose=None)
    p2 = Human(verbose=None)
    gomoku = Gomoku((p1, p2), board_size=(2, 2))

    gomoku.run()
    action = GomokuAction(1, 1)
