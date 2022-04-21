from time import sleep

import torch.cuda

import GomokuLib

import cProfile, pstats

"""
    Notes:
        ENELVER LE ***** DE GOMOKUACTION -> C'est juste un tuple !
        Reset mcts data dans le RL ? Opti ?

    TODO:
        Rename GameEngine -> Engine

        BOT :
                self.policy, self.action = self.algo(self.engine)
                return self.action
            TO :    return self.algo(self.engine)[0]

        Afficher une map 'statististique' sur les samples
            Pour voir la répartition des games sur la map

        Train sans les regles + verif que le train marche bien

            mettre les captures !
        Faire un config file avec toute les constantes du RL
        coef sur la policy du network pour le debut
        
        afficher nbr de game + dataset size + nbr model/training loop + nbr de best model
        
        Conflict between self.end_game and GameEndingCapture rule in expansion/ucb

        # Save pruning masks in state_data
        Passer le model loading dans le ModelInterface et pas dans l'Agent !
            Sinon on peut pas load un model pour le jouer avec un MCTSIA classique 
        
        # Pour améliorer le train du model:
        #     Save le path du mcts lorsque le game engine certifie une victoire.
        #     Pour qu'au tour suivant le mcts check si ce path est toujours valable.
        #     En effet si un coup random est jouer ailleurs le mcts n'aura plus du tout connaissance de ce path 
        
        Bouton pour switch entre la policy du model et les qualities du MCTS

        Graph du winrate OMG 

    To talk:
    
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device selected: {device}")

def duel():

    engine = GomokuLib.Game.GameEngine.GomokuGUI(rules=[])

    # agent = GomokuLib.AI.Agent.GomokuAgent(
    #     mcts_iter=100,
    #     mcts_hard_pruning=True,
    #     device=device
    # )
    #
    # p1 = agent
    p1 = GomokuLib.Player.RandomPlayer()
    # p2 = GomokuLib.Player.RandomPlayer()
    p2 = GomokuLib.Player.Human()

    winner = engine.run([p2, p1])  # White: 0 / Black: 1

    print(f"Winner is {winner}")

def RLtest():

    agent = GomokuLib.AI.Agent.GomokuAgent(
        GomokuLib.Game.GameEngine.GomokuGUI(rules=[]),
        mcts_iter=10,
        mcts_hard_pruning=True,
        heuristic_boost=True,
        device=device,
    )
    agent.model_comparison_mcts_iter = 10
    agent.evaluation_n_games = 1
    agent.training_loop(n_loops=2, tl_n_games=1, epochs=1)

def RLmain():

    agent = GomokuLib.AI.Agent.GomokuAgent(
        GomokuLib.Game.GameEngine.GomokuGUI(rules=[]),
        # agent_name="agent_20:04:2022_21:00:05",
        mcts_iter=100,
        # mcts_pruning=True,
        mcts_hard_pruning=True,
        heuristic_boost=True,
        device=device,
    )

    profiler = cProfile.Profile()
    profiler.enable()

    agent.training_loop(n_loops=1, tl_n_games=10, epochs=10)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('tmp_profile_from_script.prof')

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

    gameEngine = GomokuLib.Game.GameEngine.GomokuGUI(None, 19)
    agentEngine = GomokuLib.Game.GameEngine.Gomoku(None, 19)

    # mcts = GomokuLib.Algo.MCTSAI(model_interface, pruning=True, iter=1000)
    p1 = GomokuLib.AI.Agent.GomokuAgent(
        agentEngine,
        mcts_iter=100,
        # mcts_pruning=True,
        mcts_hard_pruning=True
    )
    # p1 = GomokuLib.Player.RandomPlayer()

    p2 = GomokuLib.AI.Agent.GomokuAgent(
        agentEngine,
        agent_name="agent_19:04:2022_17:15:09",
        mcts_iter=100,
        # mcts_pruning=True,
        mcts_hard_pruning=True,
        # mean_forward=True,
    )

    win_rate = 0
    n_games = 3
    for i in range(n_games):
        winner = gameEngine.run([p2, p1])  # White: 0 / Black: 1
        print(f"Players ->\np1: {p1}\np2: {p2}")
        print(f"Game {i}: winner is\n -> {winner}")
        if winner == p2:
            win_rate += 1
    print(f"last version win rate: {win_rate / n_games}")

if __name__ == '__main__':
    # duel()
    RLmain()
    # RLtest()
    # save_load_tests()
    # random_test()
    # agents_comparaison()
