from tkinter import *

from GomokuLib.Game.GameEngine.GomokuGUI import GomokuGUI
from GomokuLib.Player.Human import Human

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
    p1 = Human()
    p2 = Human()
    engine = GomokuGUI((p1, p2), 19)
    winner = engine.run()
    print(f"Winner is {winner}")


if __name__ == '__main__':
    main()