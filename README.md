# Gomoku

A Gomoku game, with an AI player.

Gomoku is a strategy board game traditionally played on a Go board with black and white stones. Two players take turns placing their stones on an intersection of the 19x19 board. A player wins by aligning 5 or more stones.

## Developper setup

Setuping Anaconda and creating environnement

```bash
./setup_dev.sh
conda activate gomoku_env
conda install --file requirements.txt
make -C GomokuLib
```

## gomoku.py usage

Optional arguments:
    * -h, --help                      Show this help message and exit
    * -p1 {mcts,pymcts,random,human}  Player 1 type
    * -p2 {mcts,pymcts,random,human}  Player 2 type
    * -p1_iter P1_ITER                Bot 1: Number of MCTS iterations
    * -p2_iter P2_ITER                Bot 2: Number of MCTS iterations
    * -p1_time P1_TIME                Bot 1: Time allowed for one turn of Monte-Carlo, in milli-seconds
    * -p2_time P2_TIME                Bot 2: Time allowed for one turn of Monte-Carlo, in milli-seconds
    * -games GAMES                    Number of games requested at the Gomoku Runner


Optional flags:
    --disable-GUI               Disable potential connection with an user interface
    --disable-Capture           Disable gomoku rule 'Capture'
    --disable-GameEndingCapture Disable gomoku rule 'GameEndingCapture'
    --disable-NoDoubleThrees    Disable gomoku rule 'NoDoubleThrees'
    --onlyUI                    Only start the User Interface
    --enable-UI                 Start the User Interface
    --host HOST                 Ip address of machine running GomokuGUIRunner
    --port PORT                 An avaible port of machine running GomokuGUIRunner
    --win_size WIN_SZ WIN_SZ    Set the size of the window: width & height


*Main script 'gomoku.py' can start the 2 following main programs :*

    * Gomoku runners program (Launched automatically):
    * User Interface program (Not launched automatically):
        `pyton hon gomoku.py --onlyUI`
        `pythgomoku.py --enable-UI ...`

*Players :*

    * human :   Allow Humans to play with the mouse on the User Interface (Only work with an active UI)
    * mcts  :   Monte-Carlo Tree Search algorithm write in Python and mostly compile in C thanks to Numba library
    * pymcts:   Monte-Carlo Tree Search algorithm write in Python
    * random:   Random plays made each turns
