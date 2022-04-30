#include "algo.h"

static void count_align(char *board, int ar, int ac, int rmax, int cmax, char *align)
{
    static int  direction[8] = {-1, -1, -1, 0, -1, 1, 0, -1};

    // four directions
    for (int direction_idx = 0; direction_idx < 8; direction_idx += 2)
    {
        // slide 4 times
        int     dr = direction[direction_idx];
        int     dc = direction[direction_idx + 1];
        int     r1 = ar;
        int     c1 = ac;
        int     count1 = 4;

        for (int i = 0; i < 4 && count1 == 4; ++i)
        {
            r1 += dr;
            c1 += dc;
            if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax || board[r1 * cmax + c1] == 0)
                count1 = i;
        }

        if (count1 < 2)         // Less than 3 stones align
            continue ;
        else if (count1 >= 4)   // 5 or more stones align
            align[2]++;

        else if (count1 == 3)   // 3 or 4 stones align in this direction -> Need to check if 2 consecutives cells are empty
        {
            r1 += dr;   // Check if next cell is empty
            c1 += dc;
            if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax || board[361 + r1 * cmax + c1] == 1)
                continue
            r1 = ar - dr;   // Check if previous cell is empty
            c1 = ac - dc;
            if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax || board[361 + r1 * cmax + c1] == 1)
                continue

            if (count1 == 2)  // 3 stones align with 2 empty cells. Need one more.
            {
                r1 -= dr;   // Check if second cell before is empty
                c1 -= dc;
                if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax ||
                    board[r1 * cmax + c1] == 1 || board[361 + r1 * cmax + c1] == 1)
                    continue
                r1 = ar + 4 * dr;   // Check if second cell after is empty
                c1 = ac + 4 * dc;
                if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax ||
                    board[r1 * cmax + c1] == 1 || board[361 + r1 * cmax + c1] == 1)
                    continue
                align[0]++;
            }
            else  // 4 stones align with 2 empty cells to win
                align[1]++;
        }
    }
}



float mcts_eval_heuristic(char *board, int *c_align_count)
{
    int     board_size = 361;
    int     i_end = 722;

    for (int i = 0; i < i_end; ++i)
    {
        if (board[i])
        {
            char        align[3] = {0, 0, 0};

        }
    }
    return 0.;
}