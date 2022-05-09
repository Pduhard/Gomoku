#include "algo.h"
#include <stdio.h>

static void count_align(char *board, char *full_board, int p_id, int ar, int ac, char *align)
{
    /*
        Count aligns:                   01|2345
            5 stones ->                   #####
            4 stones + 2 empty cells ->  _####_
            3 stones + 3 empty cells -> __###_
            3 stones + 3 empty cells ->  _###__
    */
    static int  direction[8] = {-1, 1, 0, 1, 1, 1, 1, 0};
    static int  rmax = 19, cmax = 19;
    int         lineidx[6];   // Idx of each cell on the line (Implicite action on the middle between index 3 and 4)
    char        map_edges[6]; // Bool values 4 cells in 2 directions (Implicite action on the middle between index 3 and 4)

    // four directions
    for (int direction_idx = 0; direction_idx < 8; direction_idx += 2)
    {
        // slide in this direction
        int     dr = direction[direction_idx];
        int     dc = direction[direction_idx + 1];

        int     r1 = ar - 2 * dr;
        int     c1 = ac - 2 * dc;

        for (int i = 0; i < 6; ++i) // 2 for avec un break selon map edge (init map edge a 1)
        {
            if (r1 != ar || c1 != ac)
            {
                map_edges[i] = r1 < 0 || rmax <= r1 || c1 < 0 || cmax <= c1;
                lineidx[i] = map_edges[i] ? -1 : p_id + r1 * cmax + c1;
//                fprintf(stderr, "lineid %d = %d %d |\tedge=%d |\tboard=%d |\tfull_board=%d\n", i, r1, c1, map_edges[i], map_edges[i] ? -1 : board[lineidx[i]], map_edges[i] ? -1 : full_board[lineidx[i] - p_id]);
            }
            else
                i -= 1;
            r1 += dr;
            c1 += dc;
        }

        if (map_edges[4]) // Need a 4th cell (Implicite check of id 2 and 3)
            continue ;
        if (board[lineidx[2]] == 0 || board[lineidx[3]] == 0 || // At least 3 stone align and no stone before
            (map_edges[1] == 0 && board[lineidx[1]] == 1))
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
                else if (map_edges[1] == 0 && full_board[lineidx[1] - p_id] == 0 && full_board[lineidx[5] - p_id] == 0)   // 4 stone align with 2 empty cell on side: _####_
                {
                    align[1]++;
//                    fprintf(stderr, "4 aligns: %d %d %d\n", map_edges[1] == 0, full_board[lineidx[1] - p_id] == 0, full_board[lineidx[5] - p_id] == 0);
                }
            }
            else
                continue ;
        }
        else if (full_board[lineidx[4] - p_id] == 0)    // Only 3 stones align in this direction (4th cell is empty)
        {
            if (map_edges[1] || full_board[lineidx[1] - p_id] != 0)     // Previous cell need to be valid and empty
                continue ;
            if ((map_edges[0] == 0 && full_board[lineidx[0] - p_id] == 0) || (map_edges[5] == 0 && full_board[lineidx[5] - p_id] == 0)) // Another empty cell on one side ?
            {
                align[0]++;
                // fprintf(stderr, "3 aligns: %d %d\n", map_edges[0] == 0 && full_board[lineidx[0]] == 0, map_edges[5] == 0 && full_board[lineidx[5]] == 0);
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
    for (int r = gz_start_r; r < gz_end_r; r++)
        for (int c = gz_start_c; c < gz_end_c; c++)
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
                count_align(board, full_board, 0, r, c, align_1);
//                 fprintf(stderr, "board[0, %d, %d] update to %d | %d | %d\n", r, c, align_1[0], align_1[1], align_1[2]);
            }
            else if (board[361 + tmp_i])
            {
                count_align(board, full_board, 361, r, c, align_2);
//                 fprintf(stderr, "board[1, %d, %d] update to %d | %d | %d\n", r, c, align_2[0], align_2[1], align_2[2]);
            }
        }
    return (cap_1 * cap_1 - cap_2 * cap_2) / 10. + \
        1 * (align_1[0] - align_2[0]) + \
        2.5 * (align_1[1] - align_2[1]) + \
        6 * (align_1[2] - align_2[2]);
}
