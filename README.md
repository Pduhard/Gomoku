# GOMOKU

## Overview

A Gomoku game, with Humans or Monte-Carlo Tree Search algorithm as players.

Gomoku is a strategy board game traditionally played on a Go board with black and white stones. Two players take turns placing their stones on an intersection of the 19x19 board. A player wins by aligning at least 5 stones or by making 5 captures (10 stones removed).

## Developper setup

Setuping Anaconda and creating environnement

```bash
./setup_dev.sh
conda activate gomoku_env
conda install --file requirements.txt
make -C GomokuLib
```

## gomoku.py usage

Main script 'gomoku.py' can start the 2 following main programs :

    * Gomoku runners program (Launched automatically):
    * User Interface program (Not launched automatically):

Optional arguments:
```
  -h, --help                      Show this help message and exit
  -p1 {mcts,pymcts,random,human}  Player 1 type (Default as human)
  -p2 {mcts,pymcts,random,human}  Player 2 type (Default as mcts)
  -p1_iter P1_ITER                Bot 1: Number of MCTS iterations
  -p2_iter P2_ITER                Bot 2: Number of MCTS iterations
  -p1_time P1_TIME                Bot 1: Time allowed for one turn of Monte-Carlo, in milli-seconds
  -p2_time P2_TIME                Bot 2: Time allowed for one turn of Monte-Carlo, in milli-seconds
  -games GAMES                    Number of games requested at the Gomoku Runner
```

Optional flags:
```
  --disable-GUI               Disable potential connection with an user interface
  --disable-Capture           Disable gomoku rule 'Capture'
  --disable-GameEndingCapture Disable gomoku rule 'GameEndingCapture'
  --disable-NoDoubleThrees    Disable gomoku rule 'NoDoubleThrees'
  --onlyUI                    Only start the User Interface
  --enable-UI                 Start the User Interface
  --host HOST                 Ip address of machine running GomokuGUIRunner
  --port PORT                 An avaible port of machine running GomokuGUIRunner
  --win_size WIN_SZ WIN_SZ    Set the size of the window: width & height
```

Players :
```
    human :   Allow Humans to play with the mouse on the User Interface (Only work with an active UI)
    mcts  :   Monte-Carlo Tree Search algorithm write in Python and mostly compile in C thanks to Numba library
    pymcts:   Monte-Carlo Tree Search algorithm write in Python
    random:   Random plays made each turns
```

## Monte-Carlo Tree Search implementation
(WIP)

Selection:
  Dynamic tests of valid action ...
  Pruning depends on tree depth ...
  Policy formula used:
    ...

Simulation:
  Heuristic -> Paterns rewarded in a graph ...
