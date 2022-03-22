import GomokuLib

import cProfile, pstats

"""
    Notes:

    TODO:
        Rename GameEngine -> Engine

"""

def main():

    model_interface = GomokuLib.AI.Model.ModelInterface(
        GomokuLib.AI.Model.GomokuModel(17, 19, 19),
        GomokuLib.AI.Dataset.Compose([
            GomokuLib.AI.Dataset.ToTensorTransform(),
            GomokuLib.AI.Dataset.AddBatchTransform()
        ])
    )

    p1 = GomokuLib.Player.RandomPlayer()

    mcts = GomokuLib.Algo.MCTSAI(model_interface)
    mcts.mcts_iter = 100
    p2 = GomokuLib.Player.Bot(mcts)

    engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)
    winner = engine.run([p1, p2])  # White: 0 / Black: 1

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