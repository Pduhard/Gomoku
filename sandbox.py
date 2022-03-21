from tkinter import *

from GomokuLib.Game.GameEngine.GomokuGUI import GomokuGUI
from GomokuLib.Player.Human import Human

from GomokuLib.Player.Bot import Bot
from GomokuLib.Player.RandomPlayer import RandomPlayer

from GomokuLib.Algo.MCTS import MCTS
from GomokuLib.Algo.MCTSLazy import MCTSLazy
from GomokuLib.Algo.MCTSAMAFLazy import MCTSAMAFLazy
from GomokuLib.Algo.MCTSAI import MCTSAI

from GomokuLib.AI.Model.GomokuModel import GomokuModel

import cProfile, pstats

"""

    Notes:

    TODO:
        Rename GameEngine -> Engine
        2 BasicRule dans rules_fn

"""

def main():

    # mcts = MCTS()
    # mcts.mcts_iter = 100
    # p1 = Bot(mcts)
    p1 = RandomPlayer()

    mcts = MCTSAI(GomokuModel(17, 19, 19))
    # mcts = MCTSAMAFLazy()
    mcts.mcts_iter = 100
    p2 = Bot(mcts)

    engine = GomokuGUI(None, 19, history_size=8)
    winner = engine.run([p1, p2]) # White: 0 / Black: 1

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