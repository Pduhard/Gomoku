#include "algo.h"
#include <stdio.h>

static void count_align(char *board, int p1, int p2, int ar, int ac, char *align)
{
    static int  direction[8] = {-1, 1, 0, 1, 1, 1, 1, -1};
    static int  rmax = 19, cmax = 19;

    // four directions
    for (int direction_idx = 0; direction_idx < 8; direction_idx += 2)
    {
        // slide in this direction
        int     dr = direction[direction_idx];
        int     dc = direction[direction_idx + 1];
        int     r1 = ar;
        int     c1 = ac;
        int     count1 = 4;

        // slide at least 4 times
        for (int i = 0; i < 4 && count1 == 4; i++)
        {
            r1 += dr;
            c1 += dc;
            if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax ||
                board[p1 + r1 * cmax + c1] == 0 || board[p2 + r1 * cmax + c1] == 1)
                count1 = i;
        }

        if (count1 < 2)         // Less than 3 stones align
            continue ;
        else if (count1 == 4)   // 5 or more stones align
            align[2]++;

        else   // 3 or 4 stones align in this direction -> Need to check if 2 consecutives cells are empty
        {
            // fprintf(stderr, "%d stones align %d %d on line dy=%d|dx=%d\n", count1 + 1, r1, c1, dr, dc);

//            if (count1 != 3)
//            {
//                r1 += dr;   // Check if next cell is empty
//                c1 += dc;
//            }
            if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax ||
                board[p1 + r1 * cmax + c1] == 1 || board[p2 + r1 * cmax + c1] == 1)
                continue ;
            r1 = ar - dr;   // Check if previous cell is empty
            c1 = ac - dc;
            if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax ||
                board[p1 + r1 * cmax + c1] == 1 || board[p2 + r1 * cmax + c1] == 1)
                continue ;

            if (count1 == 2)  // 3 stones align with 2 empty cells. Need one more.
            {
                // fprintf(stderr, "%d stones align %d %d on line dy=%d|dx=%d (Check at least one more empty cell)\n", count1 + 1, r1, c1, dr, dc);
                int r2 = ar + 4 * dr;       // Check if second cell after is empty
                int c2 = ac + 4 * dc;
                r1 -= dr;   // Check if second cell before is empty
                c1 -= dc;
                if ((r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax || board[p2 + r1 * cmax + c1] == 1) &&
                    (r2 < 0 || r2 >= rmax || c2 < 0 || c2 >= cmax || board[p2 + r2 * cmax + c2] == 1))
                    continue ;
                align[0]++;
            }
            else  // 4 stones align with 2 empty cells to win
                align[1]++;
        }
    }
}

float mcts_eval_heuristic(char *board, int cap_1, int cap_2)
{
    float   cap_val = (cap_2 * cap_2 - cap_1 * cap_1) / 10.;
    char    align_1[3] = {0, 0, 0};
    char    align_2[3] = {0, 0, 0};
    int     board_size = 361;
    int     i_end = 722;

    for (int i = 0; i < board_size; i++)
    {
        if (board[i])
        {
            count_align(board, 0, 361, i / 19, i % 19, align_1);
            // fprintf(stderr, "board[0, %d, %d] update to %d | %d | %d\n", i / 19, i % 19, align_1[0], align_1[1], align_1[2]);
        }
    }
    for (int i = 0, board_i = board_size; board_i < i_end; i++, board_i++)
    {
        if (board[board_i])
        {
            count_align(board, 361, 0, i / 19, i % 19, align_2);
            // fprintf(stderr, "board[1, %d, %d] update to %d | %d | %d\n", i / 19, i % 19, align_2[0], align_2[1], align_2[2]);
        }
    }
    return cap_val +\
        0.15 * (align_2[0] - align_1[0]) +\
        0.6 * (align_2[1] - align_1[1]) +\
        3 * (align_2[2] - align_1[2]);
}
