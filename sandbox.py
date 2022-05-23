from ctypes import c_buffer
import concurrent
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

        Worker
    
    
        Les captures sont dans le endturn du next_turn() donc quand
        on compute l'heuristic pour l'UI, le nbr de capture n'est pas update
            Add callbacks to next_turn
        
        Reward du mcts doit diminuer dans la backprop ?

    TODO (Important):

        
        Dupliquer l'heuristic pour valoriser les coups de l'adversaire (Qui ne sont pas les même)
            Valoriser les alignement de 4 ou il en manque qu'un !
        
        UI fonction pour afficher un board


    TODO (Pas très important):

        Faire un config file avec toute les constantes du RL
        afficher le nbr de train / epochs effectué 

        Graph du winrate OMG 
        Passer le model loading dans le ModelInterface et pas dans l'Agent !
            Sinon on peut pas load un model pour le jouer avec un MCTSIA classique 

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

    runner=GomokuLib.Game.GameEngine.GomokuGUIRunner()
    # runner=GomokuLib.Game.GameEngine.GomokuRunner()

    # runner = GomokuLib.Game.GameEngine.GomokuGUIRunner(
    #     rules=['Capture']
    # )
    # agent = GomokuLib.AI.Agent.GomokuAgent(
    #     RLengine=engine,
    #     # agent_name="agent_23:04:2022_18:14:01",
    #     agent_name="agent_13:05:2022_20:50:50",
    #     mcts_iter=1000,
    #     mcts_hard_pruning=True,
    #     mean_forward=True,
    #     model_confidence=0.95,
    #     device=device
    # )
    # p2 = agent

    # p1 = GomokuLib.Player.RandomPlayer()
    mcts_p1 = GomokuLib.Algo.MCTSEvalLazy(
        engine=runner.engine,
        iter=10000,
        hard_pruning=True,
        rollingout_turns=10
    )
    p1 = GomokuLib.Player.Bot(mcts_p1)

    mcts_p2 = GomokuLib.Algo.MCTSEvalLazy(
        engine=runner.engine,
        iter=5000,
        hard_pruning=True,
        rollingout_turns=10
    )
    p2 = GomokuLib.Player.Bot(mcts_p2)

    # p2 = GomokuLib.Player.Human()
    # p2 = GomokuLib.Player.RandomPlayer()

    if 'p2' not in locals():
        print("new p2")
        p2 = p1

    old1 = mcts_p1.mcts_iter
    old2 = mcts_p2.mcts_iter

    mcts_p1.mcts_iter = 100
    mcts_p2.mcts_iter = 100

    winner = runner.run([p1, p2])  # White: 0 / Black: 1
    mcts_p1.mcts_iter = old1
    mcts_p2.mcts_iter = old2

    # profiler = cProfile.Profile()
    # profiler.enable()

    winner = runner.run([p1, p2])  # White: 0 / Black: 1

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # # stats.print_stats()
    # stats.dump_stats('tmp_profile_from_script.prof')

    print(f"Winner is {winner}")


def RLmain():

    engine = GomokuLib.Game.GameEngine.GomokuGUIRunner(
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
    # RLtest()
    # numba_tests()
    # parrallel_test()
    # c_tests()
