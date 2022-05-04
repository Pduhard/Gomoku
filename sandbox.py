from time import sleep

import numpy as np
import torch.cuda

import GomokuLib

import cProfile, pstats

import fastcore
from fastcore._rules import ffi, lib as fastcore

"""

    TODO (Important):

        Heuristics ajouter embeter ladversaire
        Heuristics to replace value
            Test Human vs MCTS
            Create dataset with MCTS

        Test AMAF

        Nbr de best agents
        Ajouter le 1er coup random dans la comparaison
            Sinon à peut près les même game se font

        Lazy vraiment opti ? verif avec le dtime
            Si c'est possible is_valid_action est appelle a chaque fois !
            1 seule fois si jamais l'action est impossible
            Essayer de mettre 3 valeurs differentes ?
            Ne pas regenerer les gomokuAction à chaque fois !
        
        # Enlever les captures du réseau
        Save pruning masks in state_data
        
        .to(device) -> Ou les mettre ?
        Essayer de forcer lexploration en prenant au moins une fois chaque actions 

    TODO (Pas très important):

        UI
            Ajouter 'Total time'
            'dtime' -> 'turn time'
            'Total smples' ajouter le nbr total genéré depuis le debut

        Faire un config file avec toute les constantes du RL
        afficher le nbr de train / epochs effectué / nbr de best model

        Graph du winrate OMG 
        Passer le model loading dans le ModelInterface et pas dans l'Agent !
            Sinon on peut pas load un model pour le jouer avec un MCTSIA classique 

        ENELVER LE ***** DE GOMOKUACTION -> C'est juste un tuple !
        Rename GameEngine -> Engine


    A faire/essayer ?:

        Cancel les games avec plus de n coups ? boff...
    
        Les 'last_samples' sont reset à chaque nouveau model ?
            (Deja plus ou moins le cas en fonctions des parm de l'agent)
    
        Uniformiser les petites policy du réseau:
            Si pas pertinent return 0 sinon la policy

        Forcer le mcts à simuler toutes les actions possible à la depth=1 ?
            Permettrait de louper moins de choses pour mieux entrainer le model
    
        Afficher une map 'statististique' sur les samples
            Pour voir la répartition des games sur la map

        coef sur la policy du network pour le debut ?
        Filtre de 5x5 dans le CNN ? <== idée de merde

        Heuristic ne sert à rien sans les regles !

    Bug:        
        
        Conflict between self.end_game and GameEndingCapture rule in expansion/ucb


    Notes:

        Besoin urgent de favoriser l'exploration !

"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device selected: {device}")
# device = 'cpu'

def duel():

    # engine = GomokuLib.Game.GameEngine.GomokuGUI()
    engine=GomokuLib.Game.GameEngine.GomokuGUI(
            rules=['no double-threes']
    )
    # engine = GomokuLib.Game.GameEngine.GomokuGUI(rules=['Capture'])

    # agent = GomokuLib.AI.Agent.GomokuAgent(
    #     RLengine=GomokuLib.Game.GameEngine.Gomoku(rules=['Capture']),
    #     # agent_name="agent_23:04:2022_18:14:01",
    #     agent_name="agent_28:04:2022_20:39:46",
    #     mcts_iter=300,
    #     mcts_hard_pruning=True,
    #     mean_forward=True,
    #     model_confidence=0.1,
    #     device=device
    # )
    # p2 = agent

    # p1 = GomokuLib.Player.RandomPlayer()
    mcts_p1 = GomokuLib.Algo.MCTSEvalLazy(
        engine=engine,
        iter=500,
        hard_pruning=True
    )
    p1 = GomokuLib.Player.Bot(mcts_p1)
    p2 = p1

    # p2 = GomokuLib.Player.Human()
    # p2 = GomokuLib.Player.RandomPlayer()

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(10):
        winner = engine.run([p1, p2])  # White: 0 / Black: 1

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
    stats.dump_stats('tmp_profile_from_script.prof')

    print(f"Winner is {winner}")

def RLtest():

    agent = GomokuLib.AI.Agent.GomokuAgent(
        GomokuLib.Game.GameEngine.GomokuGUI(rules=['Capture']),
        # agent_name="agent_28:04:2022_20:39:46",
        mcts_iter=500,
        mcts_hard_pruning=True,
        mean_forward=False,
        device=device,
    )
    agent.evaluation_n_games = 0
    agent.model_comparison_mcts_iter = 500
    # agent.samples_per_epoch = 50
    # agent.dataset_max_length = 100

    profiler = cProfile.Profile()
    profiler.enable()

    agent.training_loop(
        nbr_tl=1,
        nbr_tl_before_cmp=1,
        nbr_games_per_tl=1,
        epochs=10
    )
    agent.save()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('tmp_profile_from_script.prof')

def RLmain():

    agent = GomokuLib.AI.Agent.GomokuAgent(
        GomokuLib.Game.GameEngine.GomokuGUI(rules=['Capture']),
        # agent_name="agent_28:04:2022_20:39:46",
        mcts_iter=1000,
        mcts_hard_pruning=True,
        mean_forward=False,
        device=device,
    )

    agent.training_loop(
        nbr_tl=-1,
        nbr_tl_before_cmp=5,
        nbr_games_per_tl=4,
        epochs=10,
        save=False
    )

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
        agent_name="agent_21:04:2022_18:19:35",
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

def c_tests():

    board = np.zeros((2, 19, 19), dtype=np.int8)
    full_board = np.zeros((19, 19), dtype=np.int8)

    board[0, 5, 6] = 1
    board[0, 5, 7] = 1
    board[0, 5, 8] = 1

    c_board = ffi.cast("char *", board.ctypes.data)
    c_full_board = ffi.cast("char *", full_board.ctypes.data)
    count = fastcore.is_double_threes(c_board, c_full_board, 5, 6)
    print(f"Count: {count}")
    breakpoint()
    # board = np.zeros((2, 19, 19), dtype=np.bool8)
    # board[1, 5, 4] = 1
    # # board[1, 5, 5] = 1
    # board[0, 5, 6] = 1
    # board[0, 5, 7] = 1
    # board[0, 5, 8] = 1
    # # board[0, 5, 9] = 1
    # board[1, 5, 10] = 1
    # c_board = ffi.cast("char *", board.ctypes.data)
    #
    # x = fastcore.mcts_eval_heuristic(c_board, 0, 0)
    # h = 1 / (1 + np.exp(-0.4 * x))
    # print(f"h = sigmoid0.4(x)")
    # print(f"{h} = sigmoid0.4({x})")

if __name__ == '__main__':
    duel()
    # RLmain()
    # RLtest()
    # agents_comparaison()
    # c_tests()