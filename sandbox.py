from time import sleep
import GomokuLib

import cProfile, pstats

"""
    Notes:
        ENELVER LE ***** DE GOMOKUACTION -> C'est juste un tuple !
        Reset mcts data dans le RL ? Opti ?

    TODO:
        Rename GameEngine -> Engine

        mettre les captures !
        
        Conflict between self.end_game and GameEndingCapture rule in expansion/ucb

        Save pruning masks in state_data
        Passer le model loading dans le ModelInterface et pas dans l'Agent !
            Sinon on peut pas load un model pour le jouer avec un MCTSIA classique 
        
        Pour améliorer le train du model:
            Save le path du mcts lorsque le game engine certifie une victoire.
            Pour qu'au tour suivant le mcts check si ce path est toujours valable.
            En effet si un coup random est jouer ailleurs le mcts n'aura plus du tout connaissance de ce path 
        
        Bouton pour switch entre la policy du model et les qualities du MCTS

        Graph du winrate OMG 

    To talk:
    
        Backpropagation -> decresing reward !?
        Filtre de 5x5 dans le CNN ?

        Avant le premier coup de la game, un agent avec 100 iter mcts simule dejà une fin de game ? NOrmal ?    
            Va trop profondement, explore pas assez
    
        Besoin urgent de favoriser l'exploration !
        Forcer le mcts à simuler toutes les actions possible à la depth=1 ?
            Permettrait de louper moins de choses pour mieux entrainer le model
    
        GomokuGUI: utiliser le meme pour le ou les agents ainsi que celui dla game ? Conflict ? (bugs visuels observé, ca a un rapport ?)
            
        Au début du train, se focus plus sur la fin de game ?
        Pertinence de l'entrainement de la value sur les premiers coups ?
            Une stone n'est pas représentatif d'une value à 1 ou -1 
        
        Lors du self-play:
            Pour chaque sample à inserer dans le dataset ->
                Inserer toutes les versions possibles (Rotations + Symétries) ?
                ainsi que d'autres sample généré à partir de celui-ci (Translations) ?
        Lors d'un play_turn():
            Pour un state, faire la moyenne des predictions de toutes les version possible du state.
            Très utile si le model est déjà bien entrainé

        Limiter le dataset aux n derniers coups (big number) et train seulement avec une partie de ces n coups
        

"""

def duel():

    engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)
    agent = GomokuLib.AI.Agent.GomokuAgent(
        engine,
        agent_name="agent_07:04:2022_16:49:51",
        mcts_iter=300,
        mcts_hard_pruning=True
    )

    # p1 = GomokuLib.Player.RandomPlayer()
    p1 = agent
    p2 = GomokuLib.Player.Human()

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
    # test_engine = GomokuLib.Game.GameEngine.Gomoku(None)

    agent = GomokuLib.AI.Agent.GomokuAgent(
        RL_engine,
        agent_name="agent_12:04:2022_16:27:46",
        mcts_iter=50,
        mcts_hard_pruning=True
    )

    rndPlayer = GomokuLib.Player.RandomPlayer()
    winners = [None] * 10
    while not all(winners[-10:]):
        try:
            agent.training_loop(n_loops=1, tl_n_games=5, epochs=10, save_all_models=False)
        except Exception as e:
            print(f"[ERROR] -> {e}")
            breakpoint()

        # print("New engine run random game to test model until it won 10 times consecutively")
        # winner = test_engine.run([rndPlayer, agent])
        # print(f"History -> {winners}")
        # print(f"Winner between RandomPlayer and RLAgent -> {winner}")
        # winners.append(winner == agent)

    print(f"This new agent win 10 times consecutively ! -> {agent}")

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

def random_test():

    engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)

    p1 = GomokuLib.AI.Agent.GomokuAgent(
        engine,
        GomokuLib.AI.Model.ModelInterface(mean_forward=True),
        agent_name="agent_07:04:2022_16:49:51",
        mcts_iter=100,
        # mcts_pruning=True
    )

    p2 = GomokuLib.Player.RandomPlayer()

    winner = engine.run([p2, p1])  # White: 0 / Black: 1
    print(f"winner is: {winner}")

def agents_comparaison():

    engine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)

    # model_interface = GomokuLib.AI.Model.ModelInterface(
    #     GomokuLib.AI.Model.GomokuModel(17, 19, 19),
    #     GomokuLib.AI.Dataset.Compose([
    #         GomokuLib.AI.Dataset.ToTensorTransform(),
    #         GomokuLib.AI.Dataset.AddBatchTransform()
    #     ])
    # )
    # mcts = GomokuLib.Algo.MCTSAI(model_interface, pruning=True, iter=1000)
    p1 = GomokuLib.AI.Agent.GomokuAgent(
        engine,
        mcts_iter=200,
        # mcts_pruning=True,
        mcts_hard_pruning=True
    )
    # p1 = GomokuLib.Player.RandomPlayer()

    p2 = GomokuLib.AI.Agent.GomokuAgent(
        engine,
        agent_name="agent_07:04:2022_16:49:51",
        mcts_iter=200,
        # mcts_pruning=True,
        mcts_hard_pruning=True,
        mean_forward=True
    )

    win_rate = 0
    n_games = 2
    for i in range(n_games):
        winner = engine.run([p2, p1])  # White: 0 / Black: 1
        print(f"Players ->\np1: {p1}\np2: {p2}")
        print(f"Game {i}: winner is\n -> {winner}")
        if winner == p2:
            win_rate += 1
    print(f"last version win rate: {win_rate / n_games}")

if __name__ == '__main__':
    # duel()
    RLmain()
    # save_load_tests()
    # random_test()
    # agents_comparaison()
