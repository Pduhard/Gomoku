from tkinter import *

from GomokuLib.Game.GameEngine.GomokuGUI import GomokuGUI
from GomokuLib.Player.Human import Human

from GomokuLib.Player.Bot import Bot

from GomokuLib.Algo.MCTS import MCTS

from GomokuLib.Game.GameEngine import Gomoku

import cProfile, pstats

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
    # engine = GomokuGUI(None, 19)

    mcts = MCTS()
    mcts.mcts_iter = 100
    p1 = Bot(mcts)
    p2 = Bot(MCTS())
    engine = GomokuGUI(None, 19)
    winner = engine.run([p2, p1])
    # winner = engine.run([Human(), Human()])
    print(f"Winner is {winner}")

    # engine = Gomoku(None, 19)
    # mcts = MCTS()
    # mcts.mcts_iter = 1000
    # action = mcts(engine, engine.state.board, engine.get_actions())
    # profiler = cProfile.Profile()
    # mcts.mcts_iter = 100
    # profiler.enable()
    # action = mcts(engine, engine.state.board, engine.get_actions())
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    # stats.dump_stats('tmp_profile_from_script.prof')
    # print(action)

if __name__ == '__main__':
    main()