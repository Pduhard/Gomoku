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
    Modif a faire pour opti:
        Enlever les full_board = board0 | board1 qui sont de partout
        if pruning.any(): à enlever dans rollingout ?


    TODO:

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

    # runner = GomokuLib.Game.GameEngine.GomokuGUIRunner()
    runner = GomokuLib.Game.GameEngine.GomokuRunner()

    # p1 = GomokuLib.Player.RandomPlayer()
    mcts_p1 = GomokuLib.Algo.MCTSNjit(
        engine=runner.engine,
        iter=1000,
        pruning=True,
        rollingout_turns=10
    )
    p1 = GomokuLib.Player.Bot(mcts_p1)

    # mcts_p2 = GomokuLib.Algo.MCTSNjit(
    #     engine=runner.engine,
    #     iter=1000,
    #     pruning=True,
    #     rollingout_turns=10
    # )
    # p2 = GomokuLib.Player.Bot(mcts_p2)

    if 'p2' not in locals():
        print("new p2")
        p2 = p1
        mcts_p2 = mcts_p1

    old1 = mcts_p1.mcts_iter
    old2 = mcts_p2.mcts_iter
    mcts_p1.mcts_iter = 100
    mcts_p2.mcts_iter = 100
    winner = runner.run([p1, p2], send_all_ss=False)  # White: 0 / Black: 1
    mcts_p1.mcts_iter = old1
    mcts_p2.mcts_iter = old2

    # p2 = GomokuLib.Player.Human()
    # p2 = GomokuLib.Player.RandomPlayer()

    # profiler = cProfile.Profile()
    # profiler.enable()

    # winner = runner.run([p1, p2])  # White: 0 / Black: 1

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
    # stats.dump_stats('tmp_profile_from_script.prof')

    print(f"Winner is {winner}")
    # breakpoint()


if __name__ == '__main__':
    duel()
