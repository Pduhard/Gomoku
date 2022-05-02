#include "rules.h"
#include <stdio.h>

int count_captures(char *board, int ar, int ac)
{
    int         count = 0;
    static int  rmax = 19, cmax = 19;
    static int  directions[16] = {
        -1, -1,
        -1, 0,
        -1, 1,
        0, 1,
        1, 1,
        1, 0,
        1, -1,
        0, -1,
    };

    // Eight directions
    for (int direction_idx = 0; direction_idx < 16; direction_idx += 2)
    {
        // Check in this direction
        int     dr = directions[direction_idx];
        int     dc = directions[direction_idx + 1];

        int     r1 = ar + 3 * dr;                                       // Second stone that made capture
        int     c1 = ac + 3 * dc;
//        fprintf(stderr, "La boucle est la ! %d %d -> %d %d\n", ar, ac, r1, c1);

        if (0 <= r1 && r1 < rmax && 0 <= c1 && c1 < cmax && board[r1 * cmax + c1] == 1)                       // Second stone that made capture
        {
            int opp_i1 = 361 + (r1 - dr) * cmax + (c1 - dc);            // Second captured stone
//            fprintf(stderr, "La boucle est semi pleine ! %d %d: opp %d %d -> opp %d / %d\n", ar, ac, r1 - dr, c1 - dc, board[opp_i1], board[opp_i1 - 361]);
            if (board[opp_i1] == 1)
            {
                int opp_i0 = 361 + (ar + dr) * cmax + (ac + dc);        // First captured stone
//                fprintf(stderr, "La boucle est tordu ? %d %d: %d %d -> %d\n", ar, ac, ar + dr, ac + dc, board[opp_i0]);
                if (board[opp_i0] == 1)
                {
//                    fprintf(stderr, "La boucle est tordu ! %d %d\n", ar, ac);
                    board[opp_i0] = 0;
                    board[opp_i1] = 0;
                    count++;
                }
            }
        }
    }
    return count;
}
