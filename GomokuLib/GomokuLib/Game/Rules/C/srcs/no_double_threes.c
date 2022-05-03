//#include "rules.h"
//#include <stdio.h>
//
///*
//    4 possible free threes with 3 possible action position
//
//       Action
//         |
//        _#_|##_
//    _##|_#_
//         |
//     __|###|_
//      _|###|__
//         |
//      _|##_|#_
//     _#|##_|_
//    _#_|##_
//    __#|##_
//         |
//     _#|_##|_
//      _|_##|#_
//        _##|_#_
//        _##|#__
//         |
//*/
//
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
//
//char is_threes(char *board, char *map_edges, char ar, char ac, char dr, char dc)
//{
//    static char rmax = 19, cmax = 19;
//    char lineidx[9];   // Idx of each cell on the line (Middle at index 4)
//    char map_edges[8]; // Bool values 4 cells in 2 directions (Implicite action on the middle between index 3 and 4)
//    char pr = ar;
//    char pc = ac;
//    char nr = ar;
//    char nc = ac;
//    lineidx[4] = ar * cmax + ac;
//    for (int i = 1; i < 5; ++i) // 2 for avec un break selon map edga
//    {
//        pr -= dr;
//        pc -= dc;
//        nr += dr;
//        nc += dc;
//        lineidx[4 - i] = pr * cmax + pc;
//        lineidx[4 + i] = nr * cmax + nc;
//        map_edges[4 - i] = pr < 0 || rmax <= pr || pc < 0 || cmax <= pc;
//        map_edges[4 - 1 + i] = nr < 0 || rmax <= nr || nc < 0 || cmax <= nc;
//        printf("line %d = %d %d | edge=%d\n", -i, lineidx[4 - i] / cmax, lineidx[4 - i] % cmax, map_edges[4 - i]);
//        printf("line %d = %d %d | edge=%d\n", +i, lineidx[4 + i] / cmax, lineidx[4 + i] % cmax, map_edges[4 - 1 + i]);
//    }
//
//    if (map_edges[3] || map_edges[4])
//        return 0;
//    else if (board[lineidx[3]])  // Previous cell ?
//    {
//        if (board[lineidx[5]])  // Next cell ?
//        {/*    _|###|__
//              __|###|_     */
//            if (board[lineidx[2]] == 0 && board[lineidx[6]] == 0 && (board[(pr - dr) * cmax + (pc - dc)] || board[nr * cmax + nc]))
//        }
//        else
//        {/*   _|##_|#_
//            _#_|##_
//             _#|##_|_
//            __#|##_       */
//
//        }
//    }
//    else
//    {
//        if (board[lineidx[5]])  // Next cell ?
//        {/*      _##|_#_
//              _#|_##|_
//               _|_##|#_
//                |_##|#__  */
//
//        }
//        else
//        {/*     _#_|##_
//            _##|_#_      */
//
//        }
//    }
//}
//
//char is_double_threes(char *board, char ar, char ac)
//{
//    /*
//        map_edges
//    */
//    static char  rmax = 19, cmax = 19;
//    int          cell_i = ar * cmax + ac;
//    char old_value = 1;
//
//    old_value ^= board[cell_i]; // Place a stone and save old value
//    board[cell_i] ^= old_value;
//    old_value ^= board[cell_i];
//
//    res
//
//    old_value ^= board[cell_i]; // Replace old value
//    board[cell_i] ^= old_value;
//    old_value ^= board[cell_i];
//    return res
//}
