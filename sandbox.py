import concurrent
import time
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

    # engine = GomokuLib.Game.GameEngine.Gomoku()
    engine=GomokuLib.Game.GameEngine.GomokuGUI()
    #         rules=['Capture']
    # )
    # engine=GomokuLib.Game.GameEngine.GomokuGUI(
    #         rules=['Capture', 'Game-Ending Capture']
    # )
    # engine = GomokuLib.Game.GameEngine.GomokuGUI(rules=['Capture'])
    #
    agent = GomokuLib.AI.Agent.GomokuAgent(
        RLengine=engine,
        agent_to_load="agent_13:05:2022_20:50:50",
        mcts_iter=1000,
        mcts_hard_pruning=True,
        mean_forward=True,
        model_confidence=0.75,
        rollingout_turns=2,
        device=device
    )
    p2 = agent

    # p1 = GomokuLib.Player.RandomPlayer()
    mcts_p1 = GomokuLib.Algo.MCTSEvalLazy(
        engine=engine,
        iter=1000,
        hard_pruning=True,
        rollingout_turns=2
    )
    p1 = GomokuLib.Player.Bot(mcts_p1)

    # mcts_p2 = GomokuLib.Algo.MCTSEvalLazy(
    #     engine=engine,
    #     iter=1000,
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

def RLmain():

    engine = GomokuLib.Game.GameEngine.GomokuGUI(
        rules=['Capture']
    )
    agent = GomokuLib.AI.Agent.GomokuAgent(
        RLengine=engine,
        # agent_to_load="agent_28:04:2022_20:39:46",
        mcts_iter=2500,
        mcts_hard_pruning=True,
        mean_forward=False,
        rollingout_turns=2,
        device=device,
    )

    agent.training_loop(
        nbr_tl=-1,
        nbr_tl_before_cmp=5,
        nbr_games_per_tl=10,
        epochs=2
    )

def c_tests():

    board = np.zeros((2, 19, 19), dtype=np.int8)
    board[1, 2, 2] = 1
    board[1, 2, 3] = 1
    board[1, 2, 4] = 1
    board[1, 2, 5] = 1
    # board[0, 2, 6] = 1
    board[0, 2, 1] = 1

    full_board = np.ascontiguousarray(board[0] | board[1]).astype(np.int8)

    c_board = ffi.cast("char *", board.ctypes.data)
    c_full_board = ffi.cast("char *", full_board.ctypes.data)
    ret = fastcore_algo.mcts_eval_heuristic(
        c_board, c_full_board,
        2, 2,
        0, 0, 18, 18
    )
    print(ret)
    # profiler = cProfile.Profile()
    # profiler.enable()

    # t = time.time()
    # for i in range(1000):
    #     full_board = np.ascontiguousarray(board[0] | board[1]).astype(np.int8)
    #     c_board = ffi.cast("char *", board.ctypes.data)
    #     c_full_board = ffi.cast("char *", full_board.ctypes.data)
    #     for r in range(19):
    #         for c in range(19):
    #             fastcore_algo.mcts_eval_heuristic(
    #                 c_board, c_full_board,
    #                 0, 0,
    #                 0, 0, 18, 18
    #             )
    # dt = time.time() - t
    # print(f"dtime={dt}")

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
    # stats.dump_stats('tmp_profile_from_script.prof')

def parrallel_test():

    engine = GomokuLib.Game.GameEngine.Gomoku()
    engine2 = GomokuLib.Game.GameEngine.Gomoku()

    mcts_p1 = GomokuLib.Algo.MCTSEvalLazy(engine=engine)
    p1 = GomokuLib.Player.Bot(mcts_p1)

    mcts_p2 = GomokuLib.Algo.MCTSEvalLazy(engine=engine2)
    p2 = GomokuLib.Player.Bot(mcts_p2)

    mcts = GomokuLib.Algo.MCTSParallel(
        engine=engine,
        # mcts_iter=1000,
        num_workers=8,
        batch_size=100,
    )


    t = time.time()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        ex.submit(engine.run, [p1, p1])
        ex.submit(engine2.run, [p2, p2])

    # ret = mcts(engine)
    dt = time.time() - t
    print(f"dtime={dt}")

if __name__ == '__main__':
    duel()
    # RLmain()
    # parrallel_test()
    # c_tests()