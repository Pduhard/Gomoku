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

            
        copy de class njit ?!

    TODO (Important):

        Dupliquer l'heuristic pour valoriser les coups de l'adversaire (Qui ne sont pas les même)
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
    engine=GomokuLib.Game.GameEngine.GomokuGUI(
            # rules=['Capture', 'Game-Ending Capture']
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
        iter=1000,
        hard_pruning=True,
        rollingout_turns=3
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


    """
    1000 iter
        Python with njit_is_align -> 26.2
        Python with C is_align    -> 2.2
        jitclass                  -> 1.4
        
    """

    engine = GomokuLib.Game.GameEngine.Gomoku()

    rule = GomokuLib.Game.Rules.BasicRuleJit()
    # rule = GomokuLib.Game.Rules.NoDoubleThreesJit()
    rule.is_valid(engine.state.board, 0, 0)
    rule.get_valid(engine.state.board)
    # rule.winning(engine.state.board, 0, 0, 0, 0, 18, 18)
    # rule.create_snapshot()
    # rule.update_from_snapshot()
    # import numba as nb
    # import fastcore._rules as md
    import ctypes
    # board.size * board.dtype.itemsize
    board = engine.state.board
    # print(cffi_board)
    # print(c_buffer)

    # caddr = ffi.addressof(c_board)
    # print(caddr)
    # c_board = ffi.cast("char *", engine.state.board.ctypes.data)
    # c_board_ffi = ffi.buffer(, board.size * board.dtype.itemsize)
    # print(c_board)



    # nb.core.typing.cffi_utils.register_module(md)

    # is_winning = md.lib.is_winning

    # def winning(board, ar, ac, gz0, gz1, gz2, gz3):

    #     return rule.winning(board, ar, ac, gz0, gz1, gz2, gz3)

    # winningjit = nb.njit(winning)
    # profiler = cProfile.Profile()
    # profiler.enable()
    rule.winning(board, np.int64(0), np.int64(5), np.int64(0), np.int64(0), np.int64(18), np.int64(18))
    n = 1000
    t = time.time()
    # for i in range(n):
    #     for a in range(19):
    #         for b in range(19):
                # rule.is_valid(engine.state.board, a, b)
                # rule.get_valid(engine.state.board)
                # print()
    rule.winning(board, np.int64(5), np.int64(5), np.int64(0), np.int64(0), np.int64(18), np.int64(18))
                # rule.winning(c_board, np.int32(a), np.int32(b), np.int32(0), np.int32(0), np.int32(18), np.int32(18))
                # rule.create_snapshot()
                # rule.update_from_snapshot()
                # rule.copy()
    dt = time.time() - t
    print(f"dtime BasiRuleJit={dt} s")
    
    print(hasattr(rule, "get_valid"))

    # rule = GomokuLib.Game.Rules.NoDoubleThrees(engine)
    rule = GomokuLib.Game.Rules.BasicRule(engine)
    # rule.winning(GomokuLib.Game.Action.GomokuAction(0, 0))

    t = time.time()
    for i in range(n):
        for a in range(19):
            for b in range(19):
                action = GomokuLib.Game.Action.GomokuAction(a, b)
                # rule.get_valid()
                # rule.is_valid(action)
                rule.winning(action)
                # rule.create_snapshot()
                # rule.update_from_snapshot(None)
                # rule.copy()
    dt = time.time() - t
    print(f"dtime BasicRule={dt} s")


    # GomokuLib.Game.Rules.njit_is_align.parallel_diagnostics(level=4)


    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # # stats.print_stats()
    # stats.dump_stats('tmp_profile_from_script.prof')

    # board[0, 0, 0] = 1
    # board[0, 2, 2] = 1
    # board[0, 3, 2] = 1
    # board[0, 4, 2] = 1
    #
    # board[1, 1, 1] = 1
    # board[1, 2, 1] = 1
    # board[1, 3, 1] = 1
    # board[1, 4, 1] = 1
    # board[0, 5, 1] = 1
    #
    # full_board = np.sum(board, axis=0).astype(np.int8)
    #
    # if not board.flags['C_CONTIGUOUS']:
    #     board = np.ascontiguousarray(board)
    #
    # if not full_board.flags['C_CONTIGUOUS']:
    #     full_board = np.ascontiguousarray(full_board)
    #
    # c_board = ffi.cast("char *", board.ctypes.data)
    # c_full_board = ffi.cast("char *", full_board.ctypes.data)
    # x = fastcore_algo.mcts_eval_heuristic(c_board, c_full_board, 0, 0, 0, 0, 18, 18)
    # h = 1 / (1 + np.exp(-0.4 * x))
    # print(f"{h} = sigmoid0.4({x})")
    # breakpoint()

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
