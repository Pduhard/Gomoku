from time import sleep
import GomokuLib

import cProfile, pstats

"""
    Notes:
        ENELVER LE ***** DE GOMOKUACTION -> C'est juste un tuple !
        Reset mcts data dans le RL ?

    TODO:
        Rename GameEngine -> Engine
        Heuristics ?..
        Prunning -> Neighboors
        mctsia reset tree ? 
        WeightedRandomSampler ? Base on abs(value) ? 
        1 itertion de mcts va si loin ! Bug ? Ou toujours les memes coups sont pris ?
"""

def main():

    model_interface = GomokuLib.AI.Model.ModelInterface(
        GomokuLib.AI.Model.GomokuModel(17, 19, 19),
        GomokuLib.AI.Dataset.Compose([
            GomokuLib.AI.Dataset.ToTensorTransform(),
            GomokuLib.AI.Dataset.AddBatchTransform()
        ])
    )


    model_interface2 = GomokuLib.AI.Model.ModelInterface(
        GomokuLib.AI.Model.GomokuModel(17, 19, 19),
        GomokuLib.AI.Dataset.Compose([
            GomokuLib.AI.Dataset.ToTensorTransform(),
            GomokuLib.AI.Dataset.AddBatchTransform()
        ])
    )

    # p1 = GomokuLib.Player.RandomPlayer()
    p1 = GomokuLib.Player.Human()

    mcts = GomokuLib.Algo.MCTSAI(model_interface)
    mcts.mcts_iter = 50
    p2 = GomokuLib.Player.Bot(mcts)


    mcts2 = GomokuLib.Algo.MCTSAI(model_interface2)
    mcts2.mcts_iter = 50
    p1 = GomokuLib.Player.Bot(mcts2)
    p2 = GomokuLib.Player.Human()

    engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)
    # i = 0
    # while True:
    #     print(i)
    #     i += 1
    #     sleep(1)
    winner = engine.run([p2, p1])  # White: 0 / Black: 1



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

def RLmain():

    # engine = GomokuLib.Game.GameEngine.Gomoku(None)
    engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)

    model_interface = GomokuLib.AI.Model.ModelInterface(
        GomokuLib.AI.Model.GomokuModel(17, 19, 19)
    )

    dataset = GomokuLib.AI.Dataset.GomokuDataset(
        GomokuLib.AI.Dataset.Compose([
            GomokuLib.AI.Dataset.HorizontalTransform(0.5),
            GomokuLib.AI.Dataset.VerticalTransform(0.5),
            GomokuLib.AI.Dataset.ToTensorTransform(),
        ])
    )

    agent = GomokuLib.AI.Agent.GomokuAgent(engine, model_interface, dataset)
    agent.set_mcts_iter(1)
    agent.trainning_loop(training_loops=1, tl_n_games=5, epochs=20)    # Make sure the network predict policies around 0 at the beginning
    agent.set_mcts_iter(200)
    agent.trainning_loop(training_loops=10, tl_n_games=2, epochs=10)


if __name__ == '__main__':
    # main()
    RLmain()
