from ctypes import c_buffer
import concurrent
import time
from time import sleep

import numba
from GomokuLib import Typing
from numba import typed, njit, prange
from numba.core import types
import numba as nb

import numpy as np
import torch.cuda
from numba.experimental import jitclass

import GomokuLib

import cProfile, pstats

import fastcore
from fastcore._rules import ffi, lib as fastcore
from fastcore._algo import lib as fastcore_algo

"""

        depth graph
        NoDoubleThrees marche plus !!!

        Probleme avec le tout dernier coup, gameEndingCapture
            Une stone peut etre reposer au meme endroit sur le 0, 0 !!

        Pourquoi des fois le server casse la connectino immediatement
            quand le client se connecte en route ...
            BrokenPipeError

        Gerer les tours en fonction d'un temps egalement

        Enlever les full_board = board0 | board1 qui sont de partout
        if pruning.any(): à enlever dans rollingout ?

    TODO:

        Clean le Typing et les truc useless dedans
        Socket: envois un signal d'arrêt afin que l'autre puisse se reco à un autre process pa rla suite


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
from GomokuLib.Algo import MCTSNjit

def getMCTSNjit(engine, amaf_policy=True):

    return MCTSNjit(
        engine=engine,
        iter=1000,
        rollingout_turns=2,
        amaf_policy=amaf_policy
    )

def duel():

    # runner = GomokuLib.Game.GameEngine.GomokuRunner()
    runner = GomokuLib.Game.GameEngine.GomokuGUIRunner()
    # runner = GomokuLib.Game.GameEngine.GomokuGUIRunner(
    #     rules=['Capture', 'Game-Ending-Capture', 'no-double-threes']
    #     # rules=[]
    # )

    # p1 = GomokuLib.Player.Human(runner)
    # mcts_p1 = getMCTSNjit(runner.engine, True)
    mcts_p1 = GomokuLib.Algo.MCTSEvalLazy(
        engine=runner.engine,
        iter=1000,
        hard_pruning=True,
        rollingout_turns=2
    )
    p1 = GomokuLib.Player.Bot(mcts_p1)

    # p2 = GomokuLib.Player.Human(runner)
    mcts_p2 = getMCTSNjit(runner.engine, True)
    # mcts_p2 = GomokuLib.Algo.MCTSEvalLazy(
    #     engine=runner.engine,
    #     iter=10000,
    #     hard_pruning=True,
    #     rollingout_turns=2
    # )
    p2 = GomokuLib.Player.Bot(mcts_p2)

    if 'p1' not in locals():
        print("new p1")
        p1 = p2
        mcts_p1 = mcts_p2
    if 'p2' not in locals():
        print("new p2")
        p2 = p1
        mcts_p2 = mcts_p1

    # old1 = mcts_p1.mcts_iter
    # old2 = mcts_p2.mcts_iter
    # mcts_p1.mcts_iter = 100
    # mcts_p2.mcts_iter = 100
    # winner = runner.run([p1, p2], send_all_ss=False)  # White: 0 / Black: 1
    # mcts_p1.mcts_iter = old1
    # mcts_p2.mcts_iter = old2

    # p2 = GomokuLib.Player.RandomPlayer()

    # profiler = cProfile.Profile()
    # profiler.enable()

    # winners = runner.run([p1, p2], n_games=1)  # White: 0 / Black: 1
    winners = runner.run([p1, p2], n_games=1)  # White: 0 / Black: 1

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
    # stats.dump_stats('tmp_profile_from_script.prof')

    print(f"Winners: {winners}")
    # breakpoint()


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


if __name__ == '__main__':
    duel()
