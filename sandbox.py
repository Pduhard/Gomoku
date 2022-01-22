from tkinter import *

from GomokuLib.Game.GameEngine.GomokuGUI import GomokuGUI
from GomokuLib.Player.Human import Human

from GomokuLib.Player.Bot import Bot

from GomokuLib.Algo.MCTS import MCTS

from GomokuLib.Game.GameEngine import Gomoku

import os
os.environ["SDL_VIDEODRIVER"]="dummy"

"""

    Notes:

    Rename GameEngine -> Engine

"""

def main():

    # root = Tk()
    # print("tk")
    # ex = GUIBoard()
    # print("gui")
    # root.geometry("1000x1000+300+300")
    # print("zgergeg")
    # root.mainloop()
    print("\n\t[SANDBOX]\n")
    # p1 = Bot(MCTS())
    # p2 = Bot(MCTS())
    # engine = GomokuGUI(None, 19)
    # print(f"Winner is {winner}")


    # engine = GomokuGUI(None, 19)
    # winner = engine.run([Human(), Bot(MCTS())])

    engine = Gomoku(None, 19)
    mcts = MCTS()
    action = mcts(engine, engine.state.board, engine.get_actions())
    print(action)

if __name__ == '__main__':
    main()