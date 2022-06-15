from ast import arg
import os
import time
import argparse

import GomokuLib

from multiprocessing import Process

players_tab = {
    'mcts': GomokuLib.Algo.MCTSNjit,
    'pymcts': GomokuLib.Algo.MCTSEvalLazy,
    'human': GomokuLib.Player.Human
}
players_str = list(players_tab.keys())

def init_runner(args):

    if args.UI:
        class_ref = GomokuLib.Game.GameEngine.GomokuGUIRunner
    else:
        class_ref = GomokuLib.Game.GameEngine.GomokuRunner

    return class_ref(
        is_capture_active=args.rule1,
        is_game_ending_capture_active=args.rule2,
        is_no_double_threes_active=args.rule3,
    )

def init_player(runner: GomokuLib.Game.GameEngine.GomokuRunner, p_str: str, p_iter: int, p_roll: int):

    if p_str == "human":
        return GomokuLib.Player.Human(runner)

    else:
        mcts = players_tab[p_str](
            engine=runner.engine,
            iter=p_iter,
            rollingout_turns=p_roll
        )
        if hasattr(mcts, 'compile'):
            print(f"\ngomoku.py: MCTSNjit: Numba compilation starting ...")
            ts = time.time()
            mcts.compile(runner.engine)
            print(f"gomoku.py: MCTSNjit: Numba compilation is finished (dtime={round(time.time() - ts, 1)})\n")
        return GomokuLib.Player.Bot(mcts)

def UI_program(args):
    gui = GomokuLib.Game.UI.UIManager(
        win_size=args.win_size,
        host=args.host[0] if args.host else None,
        port=args.port[0] if args.port else None
    )
    gui()

def duel(runner, p1, p2):
    print(f"\ngomoku.py start with:\n\tRunner: {runner}\n\tPlayer 0: {p1}\n\tPlayer 1: {p2}\n")
    runner.run([p1, p2])

def parse():
    parser = argparse.ArgumentParser(description='GomokuLib main script')
    parser.add_argument('-p1', choices=players_str, default=players_str[0], help="Player 1 type")
    parser.add_argument('-p2', choices=players_str, default=players_str[0], help="Player 2 type")

    parser.add_argument('-p1_iter', action='store', type=int, default=5000, help="Bot 1: Number of MCTS iterations")
    parser.add_argument('-p2_iter', action='store', type=int, default=5000, help="Bot 2: Number of MCTS iterations")

    parser.add_argument('-p1_roll', action='store', type=int, default=2, help="Bot 1: Number of random rollingout turns in MCTS evaluation part")
    parser.add_argument('-p2_roll', action='store', type=int, default=2, help="Bot 2: Number of random rollingout turns in MCTS evaluation part")

    parser.add_argument('--disable-UI', action='store_false', dest='UI', help="Disable gomoku user interface")
    parser.add_argument('--disable-Capture', action='store_false', dest='rule1', help="Disable gomoku rule 'Capture'")
    parser.add_argument('--disable-GameEndingCapture', action='store_false', dest='rule2', help="Disable gomoku rule 'GameEndingCapture'")
    parser.add_argument('--disable-NoDoubleThrees', action='store_false', dest='rule3', help="Disable gomoku rule 'NoDoubleThrees'")

    parser.add_argument('--host', action='store', nargs=1, default=None, type=str, help="Ip address of machine running GomokuGUIRunner")
    parser.add_argument('--port', action='store', nargs=1, default=None, type=int, help="An avaible port of machine running GomokuGUIRunner")
    parser.add_argument('--win_size', action='store', nargs=2, default=(1500, 1000), type=int, help="Set the size of the window: width & height")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    runner = init_runner(args)
    p1 = init_player(runner, args.p1, args.p1_iter, args.p1_roll)
    p2 = init_player(runner, args.p2, args.p2_iter, args.p2_roll)

    if args.UI:
        p = Process(target=UI_program, args=(args,))
        p.start()

    duel(runner, p1, p2)

    if args.UI:
        print("gomoku.py: Waiting UI ending")
        p.join()

    print("gomoku.py: OVER")
