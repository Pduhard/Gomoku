#include "algo.h"
#include <stdio.h>
#include <math.h>

static void count_align_current_player(char *board, char *full_board, int ar, int ac, char *align)
{
    /*
        Current player heuristic
        Count aligns:                   01|2345
            5 stones ->                   #####
            4 stones + 2 empty cells ->  _####_
            3 stones + 3 empty cells -> __###_
            3 stones + 3 empty cells ->  _###__
            3 stones + 3 empty cells ->  _##_#_
            3 stones + 3 empty cells ->  _#_##_

        Opponent player heuristic
        Count aligns:                   01|2345
            5 stones ->                   #####
            4 stones + 1 empty cells ->  _####X
            4 stones + 1 empty cells ->  X####_
            3 stones + 3 empty cells -> __###_
            3 stones + 3 empty cells ->  _###__
            3 stones + 3 empty cells ->  _##_#_
            3 stones + 3 empty cells ->  _#_##_
    */
    static int  direction[8] = {-1, 1, 0, 1, 1, 1, 1, 0};
    static int  rmax = 19, cmax = 19;
    static int  lineidx[6] = {0};   // Idx of each cell on the line (Implicite action on the middle between index 3 and 4)
    static char map_edges[6] = {1}; // Bool values 4 cells in 2 directions (Implicite action on the middle between index 3 and 4)

    // four directions
    for (int direction_idx = 0; direction_idx < 8; direction_idx += 2)
    {
        // slide in this direction
        int     dr = direction[direction_idx];
        int     dc = direction[direction_idx + 1];

        int     r1 = ar - dr;
        int     c1 = ac - dc;

        // lineidx initialization
        map_edges[1] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
        if (map_edges[1] == 0)
        {
            lineidx[1] = r1 * cmax + c1;

            r1 -= dr;
            c1 -= dc;
            map_edges[0] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
            if (map_edges[0] == 0)
                lineidx[0] = r1 * cmax + c1;
        }
        r1 = ar + dr;
        c1 = ac + dc;
        map_edges[2] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
        if (map_edges[2] == 0)
        {
            lineidx[2] = r1 * cmax + c1;

            r1 += dr;
            c1 += dc;
            map_edges[3] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
            if (map_edges[3] == 0)
            {
                lineidx[3] = r1 * cmax + c1;

                r1 += dr;
                c1 += dc;
                map_edges[4] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
                if (map_edges[4] == 0)
                {
                    lineidx[4] = r1 * cmax + c1;

                    r1 += dr;
                    c1 += dc;
                    map_edges[5] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
                    if (map_edges[5] == 0)
                        lineidx[5] = r1 * cmax + c1;
                }
            }
        }
        // fprintf(stderr, "lineid %d = %d %d |\tedge=%d |\tboard=%d |\tfull_board=%d\n", i, r1, c1, map_edges[i], map_edges[i] ? -1 : board[lineidx[i]], map_edges[i] ? -1 : full_board[lineidx[i] - p_id]);

        if (map_edges[4]) // Need a 4th cell (Implicite check of id 2 and 3)
            continue ;
        if (map_edges[1] == 1 || board[lineidx[1]] == 0) // No stone before
            continue ;

        if (map_edges[5] == 0 && board[361 + lineidx[5]] != 1)

        if (board[lineidx[2]] == 0 || board[lineidx[3]] == 0) // At least 3 stone align 
            continue ;

        if (board[lineidx[4]] == 1)       // 4 stone align
        {
            if (map_edges[5] == 0)              // Need a 5th cell
            {
                if (board[lineidx[5]] == 1)       // 5 stone align ! #####
                {
                    align[2]++;
                    // fprintf(stderr, "5 aligns: %d\n", full_board[lineidx[5]] == 0);
                }
                else if (map_edges[1] == 0 && full_board[lineidx[1]] == 0 && full_board[lineidx[5]] == 0)   // 4 stone align with 2 empty cell on side: _####_
                {
                    align[1]++;
//                    fprintf(stderr, "4 aligns: %d %d %d\n", map_edges[1] == 0, full_board[lineidx[1]] == 0, full_board[lineidx[5]] == 0);
                }
            }
            else
                continue ;
        }
        else if (full_board[lineidx[4]] == 0)    // Only 3 stones align in this direction (4th cell is empty)
        {
            if (map_edges[1] || full_board[lineidx[1]] != 0)     // Previous cell need to be valid and empty
                continue ;
            if ((map_edges[0] == 0 && full_board[lineidx[0]] == 0) || (map_edges[5] == 0 && full_board[lineidx[5]] == 0)) // Another empty cell on one side ?
            {
                align[0]++;
                // fprintf(stderr, "3 aligns: %d %d\n", map_edges[0] == 0 && full_board[lineidx[0]] == 0, map_edges[5] == 0 && full_board[lineidx[5]] == 0);
            }
        }
    }
}

static void count_align_opponent_player(char *board, char *full_board, int ar, int ac, char *align)
{
    /*
        X: No matters what value it is

        Current player heuristic
        Count aligns:                   01|2345
            5 stones ->                   #####
            4 stones + 2 empty cells ->  _####_
            3 stones + 3 empty cells -> __###_
            3 stones + 3 empty cells ->  _###__

            3 stones + 3 empty cells ->  _##_#_
            3 stones + 3 empty cells ->  _#_##_

        Opponent player heuristic
        Count aligns:                   01|2345
            5 stones ->                   #####
            4 stones + 1 empty cells ->  _####X
            4 stones + 1 empty cells ->  X####_
            3 stones + 3 empty cells -> __###_
            3 stones + 3 empty cells ->  _###__

            3 stones + 3 empty cells ->  _##_#_
            3 stones + 3 empty cells ->  _#_##X
            4 stones + 2 empty cells ->  X#_###X
            4 stones + 2 empty cells ->  X##_##X
            4 stones + 2 empty cells ->  X###_#X
    */
    static int  direction[8] = {-1, 1, 0, 1, 1, 1, 1, 0};
    static int  rmax = 19, cmax = 19;
    static int  lineidx[6] = {0};   // Idx of each cell on the line (Implicite action on the middle between index 3 and 4)
    static char map_edges[6] = {1}; // Bool values 4 cells in 2 directions (Implicite action on the middle between index 3 and 4)

    // four directions
    for (int direction_idx = 0; direction_idx < 8; direction_idx += 2)
    {
        // slide in this direction
        int     dr = direction[direction_idx];
        int     dc = direction[direction_idx + 1];

        int     r1 = ar - dr;
        int     c1 = ac - dc;

        map_edges[1] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
        if (map_edges[1] == 0)
        {
            lineidx[1] = r1 * cmax + c1;

            r1 -= dr;
            c1 -= dc;
            map_edges[0] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
            if (map_edges[0] == 0)
                lineidx[0] = r1 * cmax + c1;
        }

        r1 = ar + dr;
        c1 = ac + dc;
        map_edges[2] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
        if (map_edges[2] == 0)
        {
            lineidx[2] = r1 * cmax + c1;

            r1 += dr;
            c1 += dc;
            map_edges[3] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
            if (map_edges[3] == 0)
            {
                lineidx[3] = r1 * cmax + c1;

                r1 += dr;
                c1 += dc;
                map_edges[4] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
                if (map_edges[4] == 0)
                {
                    lineidx[4] = r1 * cmax + c1;

                    r1 += dr;
                    c1 += dc;
                    map_edges[5] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
                    if (map_edges[5] == 0)
                        lineidx[5] = r1 * cmax + c1;
                }
            }
        }
        // fprintf(stderr, "lineid %d = %d %d |\tedge=%d |\tboard=%d |\tfull_board=%d\n", i, r1, c1, map_edges[i], map_edges[i] ? -1 : board[lineidx[i]], map_edges[i] ? -1 : full_board[lineidx[i] - 361]);

        if (map_edges[4]) // Need a 4th cell (Implicite check of id 2 and 3)
            continue ;
        if (board[361 + lineidx[2]] == 0 || board[361 + lineidx[3]] == 0 || // At least 3 stone align and no stone before
            (map_edges[1] == 0 && board[361 + lineidx[1]] == 1))
            continue ;

        if (board[361 + lineidx[4]] == 1)       // 4 stone align
        {
            if (map_edges[5] == 0)              // Need a 5th cell
            {
                if (board[361 + lineidx[5]] == 1)       // 5 stone align ! #####
                {
                    align[2]++;
                }
                else if (map_edges[1] == 0)
                {
                    // 4 stone align with 1 empty cell on one side and no opp stone on other side: _####X ou X####_
                    if ((full_board[lineidx[1]] == 0 && board[361 + lineidx[5]] != 1) ||
                        (board[361 + lineidx[1]] != 1 && full_board[lineidx[5]] == 0))
                        align[1]++;
                }
            }
            else
                continue ;
        }
        else if (full_board[lineidx[4]] == 0)    // Only 3 stones align in this direction (4th cell is empty)
        {
            if (map_edges[1] || full_board[lineidx[1]] != 0)     // Previous cell need to be valid and empty
                continue ;
            if ((map_edges[0] == 0 && full_board[lineidx[0]] == 0) || (map_edges[5] == 0 && full_board[lineidx[5]] == 0)) // Another empty cell on one side ?
            {
                align[0]++;
            }
        }
    }
}

float mcts_eval_heuristic(char *board, char *full_board, int cap_1, int cap_2, int gz_start_r, int gz_start_c, int gz_end_r, int gz_end_c)
{
    char    align_1[3] = {0, 0, 0};
    char    align_2[3] = {0, 0, 0};
    int     tmp_i;

//    fprintf(stderr, "board: %s\n", board);
//    fprintf(stderr, "Full_board: %s\n", full_board);
    // fprintf(stderr, "\nH begin\n");
    for (int r = gz_start_r; r <= gz_end_r; ++r)
        for (int c = gz_start_c; c <= gz_end_c; ++c)
        {
            tmp_i = r * 19 + c;
//            if ((board[tmp_i] | board[361 + tmp_i]) != full_board[tmp_i])
//            {
//                fprintf(stderr, "WOOOOOOOOWWWW \n");
//                fprintf(stderr, "GZ board 0 %d %d = %d\n", r, c, board[tmp_i]);
//                fprintf(stderr, "GZ board 1 %d %d = %d\n", r, c, board[361 + tmp_i]);
//                fprintf(stderr, "GZ full_board %d %d = %d\n", r, c, full_board[tmp_i]);
//            }

            if (board[tmp_i])
            {
                count_align_current_player(board, full_board, r, c, align_1);
//                 fprintf(stderr, "board[0, %d, %d] update to %d | %d | %d\n", r, c, align_1[0], align_1[1], align_1[2]);
            }
            else if (board[361 + tmp_i])
            {
                count_align_opponent_player(board, full_board, r, c, align_2);
//                 fprintf(stderr, "board[1, %d, %d] update to %d | %d | %d\n", r, c, align_2[0], align_2[1], align_2[2]);
            }
        }
    float x = (cap_1 * cap_1 - cap_2 * cap_2) / 10. + \
        1 * align_1[0] - 2 * align_2[0] + \
        3 * align_1[1] - 5 * align_2[1] + \
        7 * align_1[2] - 9 * align_2[2];
    return 1 / (1 + exp(-0.5 * x));        // Weighted sigmoid (w=-0.4)
}

/*
    Weights:
        3: 1
        4: 2.5
        5: 6

        3: 1
        4: 2.5
        5: 6
*/


/*
    X: No matters what value it is

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
