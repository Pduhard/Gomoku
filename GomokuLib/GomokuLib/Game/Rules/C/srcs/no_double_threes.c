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

//char    get_n_align(char *board, char r, char c, char dr, char dc, char n)
//{
//    /*
//        Exclusive start at r, c until n cells visited in dr, dc direction
//    */
//    char align = 0;
//
//    for (int i = 0; i < n; ++i)
//    {
//        r += dr
//        c += dc
//        if (board[r * cmax + c])
//            align |= 1;
//        align <<= 1;
//    }
//    return align
//}

static char is_threes(char *board, char *full_board, char ar, char ac, char dr, char dc)
{
    static char rmax = 19, cmax = 19;
    int  lineidx[8];   // Idx of each cell on the line (Implicite action on the middle between index 3 and 4)
    char map_edges[8]; // Bool values 4 cells in 2 directions (Implicite action on the middle between index 3 and 4)
    char pr = ar;
    char pc = ac;
    char nr = ar;
    char nc = ac;
//    fprintf(stderr, "ar, ac | dr, dc = %d, %d | %d, %d\n", ar, ac, dr, dc);
    for (int i = 1; i < 5; ++i) // 2 for avec un break selon map edge (init map edge a 1)
    {
        pr -= dr;
        pc -= dc;
        nr += dr;
        nc += dc;
        lineidx[4 - i] = pr * cmax + pc;
        lineidx[3 + i] = nr * cmax + nc;
        map_edges[4 - i] = pr < 0 || rmax <= pr || pc < 0 || cmax <= pc;
        map_edges[3 + i] = nr < 0 || rmax <= nr || nc < 0 || cmax <= nc;
//        fprintf(stderr, "line %d = %d %d | edge=%d\n", -i, lineidx[4 - i] / cmax, lineidx[4 - i] % cmax, map_edges[4 - i]);
//        fprintf(stderr, "line %d = %d %d | edge=%d\n", +i, lineidx[3 + i] / cmax, lineidx[3 + i] % cmax, map_edges[3 + i]);
    }

    if (map_edges[3] || map_edges[4])
        return 0;
    else if (board[lineidx[3]])  // Previous cell ?
    {
        if (board[lineidx[4]])  // Next cell ?
        {/*   012 3 4 567
               _|###|__
              __|###|_     */
            return (((map_edges[1] == 0 && full_board[lineidx[1]] == 0) || (map_edges[6] == 0 && full_board[lineidx[6]] == 0)) &&
                map_edges[2] == 0 && map_edges[5] == 0 && full_board[lineidx[2]] == 0 && full_board[lineidx[5]] == 0);
        }
        else
        {/* 012 3 4 567
              _|##_|#_
             _#|##_|_
            _#_|##_|
            __#|##_|       */
            if (map_edges[2])
                return 0;           // No possible case

            if (map_edges[6] && full_board[lineidx[2]] == 0 && board[lineidx[5]] == 1 && full_board[lineidx[6]] == 0) // _|##_|#_
                return 1;
            if (map_edges[1])
                return 0;           // No more possible case

            if (map_edges[5] && full_board[lineidx[1]] == 0 && board[lineidx[2]] == 1 && full_board[lineidx[5]] == 0) // _#|##_|_
                return 1;
            if (map_edges[0])
                return 0;           // No more possible case

            if (full_board[lineidx[0]] == 0)
            {
                if (board[lineidx[1]] == 1 && full_board[lineidx[2]] == 0) // _#_|##_|
                    return 1;
                return full_board[lineidx[1]] == 0 && board[lineidx[2]] == 1; // __#|##_|
            }
        }
    }
    else
    {
        if (board[lineidx[5]])  // Next cell ?
        {/*  012 3 4 567
                |_##|_#_
              _#|_##|_
               _|_##|#_
                |_##|#__  */
            if (map_edges[5])
                return 0;           // No possible case

            if (map_edges[1] && full_board[lineidx[5]] == 0 && board[lineidx[2]] == 1 && full_board[lineidx[1]] == 0) // _#|_##|_
                return 1;
            if (map_edges[6])
                return 0;           // No more possible case

            if (map_edges[2] && full_board[lineidx[2]] == 0 && board[lineidx[5]] == 1 && full_board[lineidx[6]] == 0) // _|_##|#_
                return 1;
            if (map_edges[0])
                return 0;           // No more possible case

            if (full_board[lineidx[7]] == 0)
            {
                if (full_board[lineidx[5]] == 0 && board[lineidx[6]] == 1) // |_##|_#_
                    return 1;
                return board[lineidx[5]] == 1 && full_board[lineidx[6]] == 0; // |_##|#__
            }
        }
        else
        {/* 012 3 4 567
               |_#_|##_
            _##|_#_|      */
            return ((map_edges[0] == 0 && board[lineidx[2]] == 1 && board[lineidx[1]] == 1 && full_board[lineidx[0]] == 0) ||
                (map_edges[7] == 0 && board[lineidx[5]] == 1 && board[lineidx[6]] == 1 && full_board[lineidx[7]] == 0));
        }
    }
    return 0;
}

static char count_threes(char *board, char *full_board, int ar, int ac)
{
    char count = is_threes(board, full_board, ar, ac, -1, 1);
    count += is_threes(board, full_board, ar, ac, 0, 1);
    if (count == 2)
        return 2;
    count += is_threes(board, full_board, ar, ac, 1, 1);
    if (count == 0)
        return 0;
    if (count == 2)
        return 2;
    count += is_threes(board, full_board, ar, ac, 1, -1);
//    fprintf(stderr, "count=%d\n", count);
    return count;
}

int is_double_threes(char *board, char *full_board, int ar, int ac)
{
    char old_value = 1;
    int  cell_i = ar * 19 + ac;

    old_value ^= board[cell_i]; // Place a stone and save old value
    board[cell_i] ^= old_value;
    old_value ^= board[cell_i];

    char count = count_threes(board, full_board, ar, ac);

    old_value ^= board[cell_i]; // Replace old value
    board[cell_i] ^= old_value;
    old_value ^= board[cell_i];
    return count > 1;
}
