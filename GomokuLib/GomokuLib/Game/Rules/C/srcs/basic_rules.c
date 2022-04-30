#include "rules.h"
#include <stdio.h>

int basic_rule_winning(char *board, int ar, int ac, int rmax, int cmax)
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

        int     r2 = ar;
        int     c2 = ac;
        int     count2 = 4;

        for (int i = 0; i < 4 && count2 == 4; ++i)
        {
            r2 -= dr;
            c2 -= dc;
            if (r2 < 0 || r2 >= rmax || c2 < 0 || c2 >= cmax || board[r2 * cmax + c2] == 0)
                count2 = i;
        }

        if (count1 + count2 + 1 >= 5)
        {
            // printf("WIN HEREE %d %d\n", count1, count2);
            return 1;
        }
        // else printf("NO WIN %d %d\n", count1, count2);
    }
    return 0;
}
