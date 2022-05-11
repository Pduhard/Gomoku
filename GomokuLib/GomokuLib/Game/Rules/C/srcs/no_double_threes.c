#include "rules.h"
#include <stdio.h>

/*
    4 possible free threes with 3 possible action position

       Action
         |
        _#_|##_
    _##|_#_
         |
     __|###|_
      _|###|__
         |
      _|##_|#_
     _#|##_|_
    _#_|##_
    __#|##_
         |
     _#|_##|_
      _|_##|#_
        _##|_#_
        _##|#__
         |
*/

static int is_threes(char *board, long *full_board, long ar, long ac, long dr, long dc)
{
    static long rmax = 19, cmax = 19;
    int  lineidx[8];   // Idx of each cell on the line (Implicite action on the middle between index 3 and 4)
    int  map_edges[8]; // Bool values 4 cells in 2 directions (Implicite action on the middle between index 3 and 4)
    long pr = ar;
    long pc = ac;
    long nr = ar;
    long nc = ac;
//    fprintf(stderr, "\nar, ac | dr, dc = %d, %d | %d, %d\n", ar, ac, dr, dc);
    for (int i = 1; i < 5; ++i) // 2 for avec un break selon map edge (init map edge a 1)
    {
        pr -= dr;
        pc -= dc;
        nr += dr;
        nc += dc;
        map_edges[4 - i] = pr < 0 || rmax <= pr || pc < 0 || cmax <= pc;
        map_edges[3 + i] = nr < 0 || rmax <= nr || nc < 0 || cmax <= nc;
        lineidx[4 - i] = map_edges[4 - i] ? -1 : pr * cmax + pc;
        lineidx[3 + i] = map_edges[3 + i] ? -1 : nr * cmax + nc;
//        fprintf(stderr, "lineid %d = %d %d |\tedge=%d |\tboard=%d\n", 4-i, lineidx[4 - i] / cmax, lineidx[4 - i] % cmax, map_edges[4 - i], map_edges[4 - i] ? -1 : board[lineidx[4 - i]]);
//        fprintf(stderr, "lineid %d = %d %d |\tedge=%d |\tboard=%d\n", 3+i, lineidx[3 + i] / cmax, lineidx[3 + i] % cmax, map_edges[3 + i], map_edges[3 + i] ? -1 : board[lineidx[3 + i]]);
    }

    if (map_edges[3] || map_edges[4])
        return 0;
    else if (board[lineidx[3]])  // Previous cell ?
    {
        if (board[lineidx[4]])  // Next cell ?
        {/*   012 3 4 567
               _|###|__
              __|###|_     */
            // fprintf(stderr, "(%d && %d || %d && %d) && %d && %d && %d && %d\n", map_edges[1] == 0, full_board[lineidx[1]] == 0, map_edges[6] == 0, full_board[lineidx[6]] == 0, map_edges[2] == 0, map_edges[5] == 0, full_board[lineidx[2]] == 0, full_board[lineidx[5]] == 0);
            return (((map_edges[1] == 0 && full_board[lineidx[1]] == 0) || (map_edges[6] == 0 && full_board[lineidx[6]] == 0)) &&
                map_edges[2] == 0 && map_edges[5] == 0 && full_board[lineidx[2]] == 0 && full_board[lineidx[5]] == 0);
        }
        else if (full_board[lineidx[4]] == 0)
        {/* 012 3 4 567
              _|##_|#_
             _#|##_|_
            _#_|##_|
            __#|##_|       */
            if (map_edges[2])
                return 0;           // No possible case
            // fprintf(stderr, "Here 0 0\n");

            if (map_edges[6] == 0 && full_board[lineidx[2]] == 0 && board[lineidx[5]] == 1 && full_board[lineidx[6]] == 0) // _|##_|#_
                return 1;
            if (map_edges[1])
                return 0;           // No more possible case

            // fprintf(stderr, "Here 0 1\n");
            if (map_edges[5] == 0 && full_board[lineidx[1]] == 0 && board[lineidx[2]] == 1 && full_board[lineidx[5]] == 0) // _#|##_|_
                return 1;

            if (map_edges[0] == 0 && full_board[lineidx[0]] == 0)
            {
                // fprintf(stderr, "Here 0 2\n");
                if (board[lineidx[1]] == 1 && full_board[lineidx[2]] == 0) // _#_|##_|
                    return 1;
                // fprintf(stderr, "Here 0 3\n");
                return full_board[lineidx[1]] == 0 && board[lineidx[2]] == 1; // __#|##_|
            }
        }
    }
    else if (full_board[lineidx[3]] == 0)
    {
        if (board[lineidx[4]])  // Next cell ?
        {/*  012 3 4 567
                |_##|_#_
              _#|_##|_
               _|_##|#_
                |_##|#__  */
            if (map_edges[5])
                return 0;           // No possible case

            // fprintf(stderr, "Here 1 0\n");
            if (map_edges[1] == 0 && full_board[lineidx[5]] == 0 && board[lineidx[2]] == 1 && full_board[lineidx[1]] == 0) // _#|_##|_
                return 1;
            if (map_edges[6])
                return 0;           // No more possible case

            // fprintf(stderr, "Here 1 1 | %d %d %d %d\n", map_edges[2] == 0, full_board[lineidx[2]] == 0, board[lineidx[5]] == 1, full_board[lineidx[6]] == 0);
            if (map_edges[2] == 0 && full_board[lineidx[2]] == 0 && board[lineidx[5]] == 1 && full_board[lineidx[6]] == 0) // _|_##|#_
                return 1;

            if (map_edges[7] == 0 && full_board[lineidx[7]] == 0)
            {
                // fprintf(stderr, "Here 1 2\n");
                if (full_board[lineidx[5]] == 0 && board[lineidx[6]] == 1) // |_##|_#_
                    return 1;

                // fprintf(stderr, "Here 1 3\n");
                return board[lineidx[5]] == 1 && full_board[lineidx[6]] == 0; // |_##|#__
            }
        }
        else if (full_board[lineidx[4]] == 0)
        {/* 012 3 4 567
               |_#_|##_
            _##|_#_|      */
            // fprintf(stderr, "(%d && %d && %d && %d) || (%d && %d && %d && %d)\n", map_edges[0] == 0, board[lineidx[2]] == 1, board[lineidx[1]] == 1, full_board[lineidx[0]] == 0, map_edges[7] == 0, board[lineidx[5]] == 1, board[lineidx[6]] == 1, full_board[lineidx[7]] == 0);

            return ((map_edges[0] == 0 && board[lineidx[2]] == 1 && board[lineidx[1]] == 1 && full_board[lineidx[0]] == 0) ||
                (map_edges[7] == 0 && board[lineidx[5]] == 1 && board[lineidx[6]] == 1 && full_board[lineidx[7]] == 0));
        }
    }
    return 0;
}

static char count_threes(char *board, long *full_board, long ar, long ac)
{
    int count = is_threes(board, full_board, ar, ac, -1, 1);
    // fprintf(stderr, "count0=%d\n", count);

    count += is_threes(board, full_board, ar, ac, 0, 1);
    // fprintf(stderr, "count1=%d\n", count);
    if (count == 2)
        return 2;
    count += is_threes(board, full_board, ar, ac, 1, 1);
    // fprintf(stderr, "count2=%d\n", count);
    if (count == 0)
        return 0;
    if (count == 2)
        return 2;
    count += is_threes(board, full_board, ar, ac, 1, 0);
    // fprintf(stderr, "count3=%d\n", count);
    return count;
}

int is_double_threes(char *board, long *full_board, long ar, long ac)
{
    int old_value = 1;
    long  cell_i = ar * 19 + ac;

    old_value ^= board[cell_i]; // Place a stone and save old value
    board[cell_i] ^= old_value;
    old_value ^= board[cell_i];

    int count = count_threes(board, full_board, ar, ac);

    old_value ^= board[cell_i]; // Replace old value
    board[cell_i] ^= old_value;
    old_value ^= board[cell_i];
    return count > 1;
}
