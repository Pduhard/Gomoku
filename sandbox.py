from time import sleep

import numpy as np
import torch.cuda

import GomokuLib

import cProfile, pstats

import fastcore
from fastcore._rules import ffi, lib as fastcore
from fastcore._algo import lib as fastcore_algo

"""

    Today :
    
        rollingout pendant levaluation
        Opti rollingout
        astype dans l'heuristic
        Debug capture envoye a lheuristic
        Upgrade color UI
        
        Les captures sont dans le endturn du next_turn() donc quand
        on compute l'heuristic pour l'UI, le nbr de capture n'est pas update
            Add callbacks to next_turn
        
        Reward du mcts doit diminuer dans la backprop ?

    TODO (Important):

        Dupliquer l'heuristic pour valoriser les coups de l'adversaire (Qui ne sont pas les même)
        UI fonction pour afficher un board 

    TODO (Pas très important):

        Faire un config file avec toute les constantes du RL
        afficher le nbr de train / epochs effectué / nbr de best model

        Graph du winrate OMG 
        Passer le model loading dans le ModelInterface et pas dans l'Agent !
            Sinon on peut pas load un model pour le jouer avec un MCTSIA classique 

        ENELVER LE ***** DE GOMOKUACTION -> C'est juste un tuple !
        Rename GameEngine -> Engine


    A faire/essayer ?:

        Uniformiser les petites policy du réseau:
            Si pas pertinent return 0 sinon la policy

        Afficher une map 'statististique' sur les samples
            Pour voir la répartition des games sur la map

        coef sur la policy du network pour le debut ?
        Filtre de 5x5 dans le CNN ? <== idée de merde

    Bug:        
        

    Notes:


"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device selected: {device}")
# device = 'cpu'

def duel():

    # engine = GomokuLib.Game.GameEngine.GomokuGUI()
    # engine=GomokuLib.Game.GameEngine.GomokuGUI(
    #         rules=['no double-threes']
    # )
    engine=GomokuLib.Game.GameEngine.GomokuGUI(
            rules=['Capture', 'Game-Ending Capture']
    )
    # engine = GomokuLib.Game.GameEngine.GomokuGUI(rules=['Capture'])
    #
    # agent = GomokuLib.AI.Agent.GomokuAgent(
    #     RLengine=engine,
    #     # agent_name="agent_23:04:2022_18:14:01",
    #     # agent_name="agent_28:04:2022_20:39:46",
    #     mcts_iter=1000,
    #     mcts_hard_pruning=True,
    #     mean_forward=True,
    #     model_confidence=0,
    #     device=device
    # )
    # p2 = agent

    # p1 = GomokuLib.Player.RandomPlayer()
    mcts_p1 = GomokuLib.Algo.MCTSEvalLazy(
        engine=engine,
        iter=3000,
        hard_pruning=True,
        rollingout_turns=2
    )
    p1 = GomokuLib.Player.Bot(mcts_p1)

    # mcts_p2 = GomokuLib.Algo.MCTSEvalLazy(
    #     engine=engine,
    #     iter=3000,
    #     hard_pruning=True,
    #     rollingout_turns=2
    # )
    # p2 = GomokuLib.Player.Bot(mcts_p2)

    # p2 = GomokuLib.Player.Human()
    # p2 = GomokuLib.Player.RandomPlayer()

    profiler = cProfile.Profile()
    profiler.enable()

    if 'p2' not in locals():
        print("new p2")
        p2 = p1
    for i in range(1):
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

def tmp():

    for i in range(100000):
        rd = np.random.random_integers(0, 1, size=(19, 19))
        c_rd = ffi.cast("char *", rd.ctypes.data)

        for y in range(19):
            for x in range(19):
                fastcore.is_winning(c_rd, y, x)
                fastcore.basic_rule_winning(c_rd, y, x)

def c_tests():

    profiler = cProfile.Profile()
    profiler.enable()

    # tmp()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
    stats.dump_stats('tmp_profile_from_script.prof')

    board = np.zeros((2, 19, 19), dtype=np.int8, order='C')
    board[0, 0, 0] = 1
    board[0, 2, 2] = 1
    board[0, 3, 2] = 1
    board[0, 4, 2] = 1

    board[1, 1, 1] = 1
    board[1, 2, 1] = 1
    board[1, 3, 1] = 1
    board[1, 4, 1] = 1
    board[0, 5, 1] = 1

    full_board = np.sum(board, axis=0).astype(np.int8)

    if not board.flags['C_CONTIGUOUS']:
        board = np.ascontiguousarray(board)

    if not full_board.flags['C_CONTIGUOUS']:
        full_board = np.ascontiguousarray(full_board)

    c_board = ffi.cast("char *", board.ctypes.data)
    c_full_board = ffi.cast("char *", full_board.ctypes.data)
    x = fastcore_algo.mcts_eval_heuristic(c_board, c_full_board, 0, 0, 0, 0, 18, 18)
    h = 1 / (1 + np.exp(-0.4 * x))
    print(f"{h} = sigmoid0.4({x})")
    breakpoint()

if __name__ == '__main__':
    duel()
    # RLmain()
    # RLtest()
    # c_tests()