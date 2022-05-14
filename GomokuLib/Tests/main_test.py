import GomokuLib
from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Player.Human import Human

if __name__ == "__main__":

    p1 = Human(verbose=None)
    p2 = Human(verbose=None)

    gomoku = Gomoku((p1, p2), board_size=(2, 2))

    # gomoku.register((p1, p2))
    gomoku.run()
