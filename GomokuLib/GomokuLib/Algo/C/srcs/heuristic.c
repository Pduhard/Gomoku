/*
    _:  Empty cell
    #:  Current player stone
    X:  No matters what cell it is

    Current player heuristic
        Indexes:                        01|2345
            5 stones ->                 XX#####

            4 stones + 1 empty cells -> X_####_

            3 stones + 3 empty cells -> __###_X
            3 stones + 3 empty cells -> X_#_##_
            3 stones + 3 empty cells -> X_##_#_
            3 stones + 3 empty cells -> X_###__

    Opponent player heuristic:
        Indexes:                        01|2345
            5 stones ->                 XX#####

            4 stones + 1 empty cells -> X_####X
            4 stones + 1 empty cells -> XX#_###
            4 stones + 1 empty cells -> XX##_##
            4 stones + 1 empty cells -> XX###_#
            4 stones + 1 empty cells -> XX####_

            3 stones + 3 empty cells -> __###_X
            3 stones + 3 empty cells -> X_#_##_
            3 stones + 3 empty cells -> X_##_#_
            3 stones + 3 empty cells -> X_###__
*/

// #include <math.h>

