from ctypes import c_buffer
import time
from time import sleep

import numba
from numba import typed, njit, prange
from numba.core import types

import numpy as np
import torch.cuda
from numba.experimental import jitclass

import GomokuLib

import cProfile, pstats

import fastcore
from fastcore._rules import ffi, lib as fastcore
from fastcore._algo import lib as fastcore_algo

"""

    Today :

        NodoubleThrees

    TODO (Important):

        Essayer de ne pas recréer de relge dans le update
        
        Dupliquer l'heuristic pour valoriser les coups de l'adversaire (Qui ne sont pas les même)
        Valoriser les alignement de 4 ou il en manque qu'un !
        
        UI fonction pour afficher un board 

    TODO (Pas très important):

        Faire un config file avec toute les constantes du RL
        afficher le nbr de train / epochs effectué 

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
    engine=GomokuLib.Game.GameEngine.GomokuGUI()
    # engine=GomokuLib.Game.GameEngine.GomokuGUI(
            # rules=['Capture', 'Game-Ending Capture']
    # )
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

    mcts_p2 = GomokuLib.Algo.MCTSEvalLazy(
        engine=engine,
        iter=3000,
        hard_pruning=True,
        rollingout_turns=2
    )
    p2 = GomokuLib.Player.Bot(mcts_p2)

    # p2 = GomokuLib.Player.Human()
    # p2 = GomokuLib.Player.RandomPlayer()

    if 'p2' not in locals():
        print("new p2")
        p2 = p1

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(1):
        winner = engine.run([p1, p2])  # White: 0 / Black: 1

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
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


    """
    1000 iter
        Python with njit_is_align -> 26.2
        Python with C is_align    -> 2.2
        jitclass                  -> 1.4
        
    """

    a = np.array([[
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
        dtype=np.int8,
        order='C')

    a = np.ascontiguousarray(a)


    full_a = np.int64(a[0] | a[1])
    full_a = np.ascontiguousarray(full_a)

    c_board = ffi.cast("char *", a.ctypes.data)
    c_full_board = ffi.cast("long *", full_a.ctypes.data)

    print(f"All True: {[fastcore.is_double_threes(c_board, c_full_board, 4, 4) for i in range(100)]}")

    exit(0)


    engine = GomokuLib.Game.GameEngine.Gomoku()

    rule = GomokuLib.Game.Rules.NoDoubleThrees(engine)
    ruleJit = GomokuLib.Game.Rules.NoDoubleThreesJit(engine.state.board)

    for i in range(1000):
        board = np.random.randint(0, 2, (2, 19, 19))
        engine.state.board = board
        for a in range(19):
            for b in range(19):

                action = GomokuLib.Game.Action.GomokuAction(a, b)
                full_board = board[0] | board[1]

                # rj_ret = ruleJit.get_valid(full_board)
                # r_ret = rule.get_valid()
                # if np.any(rj_ret != r_ret):
                #     print(f"ERROR get_validJit: {a}\n{b}")
                #     breakpoint()

                rj_ret = ruleJit.is_valid(full_board, a, b)
                r_ret = rule.is_valid(action)
                if np.any(rj_ret != r_ret):
                    print(f"ERROR get_validJit: {a}\n{b}")
                    breakpoint()

                # # ruleJit.winning(np.int64(0), np.int64(5), np.int64(0), np.int64(0), np.int64(18), np.int64(18))
                # rj_ret = ruleJit.winning(a, b, *engine.game_zone)
                # r_ret = rule.winning(action)
                # if rj_ret != r_ret:
                #     print(f"ERROR get_validJit: {a}\n{b}")
                #     breakpoint()

        print(f"Valid board: {board}")


    # t = time.time()
    # dt = time.time() - t
    # print(f"dtime BasicRule={dt} s")

    # profiler = cProfile.Profile()
    # profiler.enable()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # # stats.print_stats()
    # stats.dump_stats('tmp_profile_from_script.prof')

def numba_tests():
    spec = [
        ('d', numba.typeof(typed.Dict.empty(types.string, types.int64))),
    ]

    @njit(parallel=True, nogil=True)
    def par(d, value):
        for i in prange(100000):
            d[str(hash(str(i)))]
            d[str(i) * 722]

    @jitclass(spec)
    class A:
        def __init__(self):
            self.d = typed.Dict.empty(types.string, types.int64)

        def run(self, value):
            par(self.d, value)

    class B:
        def __init__(self):
            self.d = {}

    a = A()
    b = B()

    value = np.random.randint(10)
    a.d[str(-1) * 722] = value
    b.d[str(-1) * 722] = value
    a.run(value)

    t = time.time()
    a.run(value)
    dt = time.time() - t
    print(f"dtime {a}={dt} s")


    t = time.time()
    for i in range(100000):
        b.d[str(hash(str(i)))]
        b.d[str(i) * 722]

    dt = time.time() - t
    print(f"dtime {b}={dt} s")

if __name__ == '__main__':
    duel()
    # RLmain()
    # RLtest()
    # numba_tests()
    # c_tests()
