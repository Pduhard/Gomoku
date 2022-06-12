#include "rules.h"
#include <stdio.h>

int count_captures(char *board, int ar, int ac, int gz_start_r, int gz_start_c, int gz_end_r, int gz_end_c, int player_idx, int *captured)
{
    int         count = 0;
    static int  cmax = 19;
    int         p_idx = player_idx * 361;
    int         opp_idx = (player_idx ^ 1) * 361;
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

//        if (0 <= r1 && r1 < rmax && 0 <= c1 && c1 < cmax && board[r1 * cmax + c1] == 1)                       // Second stone that made capture
        if (gz_start_r <= r1 && r1 <= gz_end_r && gz_start_c <= c1 && c1 <= gz_end_c && board[p_idx + r1 * cmax + c1] == 1)                       // Second stone that made capture
        {
            int row1 = r1 - dr;
            int col1 = c1 - dc;
            int opp_i1 = opp_idx + row1 * cmax + col1;            // Second captured stone
//            fprintf(stderr, "La boucle est semi pleine ! %d %d: opp %d %d -> opp %d / %d\n", ar, ac, r1 - dr, c1 - dc, board[opp_i1], board[opp_i1 - 361]);
            if (board[opp_i1] == 1)
            {
                int row0 = ar + dr;
                int col0 = ac + dc;
                int opp_i0 = opp_idx + row0 * cmax + col0;        // First captured stone
//                fprintf(stderr, "La boucle est tordu ? %d %d: %d %d -> %d\n", ar, ac, ar + dr, ac + dc, board[opp_i0]);
                if (board[opp_i0] == 1)
                {
//                    fprintf(stderr, "La boucle est tordu ! %d %d\n", ar, ac);
                    board[opp_i0] = 0;
                    board[opp_i1] = 0;

                    int baseidx = count * 6;
                    captured[baseidx] = row0;
                    captured[baseidx + 1] = col0;

                    captured[baseidx + 2] = row1;
                    captured[baseidx + 3] = col1;

                    count++;
                }
            }
        }
    }
    return count;
}
