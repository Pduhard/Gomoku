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
        
        Communication entre Agent <-> UI pour epoch / n self-play games 
        
        
    To talk:
    
        POur chaque sample à inserer dans le dataset ->
            Inserer toutes les versions possibles (Rotations + Symétries)
            ainsi que d'autres sample généré à partir de celui-ci (Translations)

        Au début du train, se focus plus sur la fin de game ?
        Limiter le dataset aux n derniers coups (big number) et train seulement avec une partie de ces n coups
        

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

    RL_engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)

    dataset = GomokuLib.AI.Dataset.GomokuDataset(
        GomokuLib.AI.Dataset.Compose([
            GomokuLib.AI.Dataset.HorizontalTransform(0.5),
            GomokuLib.AI.Dataset.VerticalTransform(0.5),
            GomokuLib.AI.Dataset.ToTensorTransform(),
        ])
    )

    rndPlayer = GomokuLib.Player.RandomPlayer()
    agent = GomokuLib.AI.Agent.GomokuAgent(
        RL_engine,
        GomokuLib.AI.Model.ModelInterface(
            GomokuLib.AI.Model.GomokuModel(17, 19, 19)
        ),
        dataset, mcts_iter=50
    )

    winners = [None] * 10
    while not all(winners[-10:]):
        agent.training_loop(n_loops=2, tl_n_games=3, epochs=10)

        print("New engine run random game to test model until it won 10 times consecutively")
        winner = RL_engine.run([rndPlayer, agent])
        print(f"History -> {winners}")
        print(f"Winner between RandomPlayer and RLAgent -> {winner}")
        winners.append(winner == agent)

def save_load_tests():
    RL_engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)

    dataset = GomokuLib.AI.Dataset.GomokuDataset(
        GomokuLib.AI.Dataset.Compose([
            GomokuLib.AI.Dataset.HorizontalTransform(0.5),
            GomokuLib.AI.Dataset.VerticalTransform(0.5),
            GomokuLib.AI.Dataset.ToTensorTransform(),
        ])
    )

    rndPlayer = GomokuLib.Player.RandomPlayer()
    agent_save = GomokuLib.AI.Agent.GomokuAgent(
        RL_engine,
        GomokuLib.AI.Model.ModelInterface(
            GomokuLib.AI.Model.GomokuModel(17, 19, 19)
        ),
        dataset,
        mcts_iter=2
    )

    agent_save.training_loop(tl_n_games=1, epochs=5)
    agent_save.save_best_model(name="save_load_test_model.pt")

    agent_load = GomokuLib.AI.Agent.GomokuAgent(
        RL_engine,
        GomokuLib.AI.Model.ModelInterface(
            GomokuLib.AI.Model.GomokuModel(17, 19, 19)
        ),
        dataset,
        mcts_iter=50
    )
    agent_load.load(model_name="save_load_test_model.pt")
    print(f"Winner: {RL_engine.run([rndPlayer, agent_load])}")


if __name__ == '__main__':
    # main()
    RLmain()
    # save_load_tests()
