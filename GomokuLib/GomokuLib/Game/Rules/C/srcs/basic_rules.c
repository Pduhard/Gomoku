#include "rules.h"
#include <stdio.h>

int is_winning(char *board, int p, int ar, int ac, int gz_start_r, int gz_start_c, int gz_end_r, int gz_end_c)
{
    static int  direction[8] = {-1, -1, -1, 0, -1, 1, 0, 1};
    static int  cmax = 19;

    // four directions
    for (int direction_idx = 0; direction_idx < 8; direction_idx += 2)
    {
        // slide 4 times
        int     dr = direction[direction_idx];
        int     dc = direction[direction_idx + 1];

        int     r1 = ar;                             // Start on line center / action
        int     c1 = ac;
        int     r2 = ar - 4 * dr;
        int     c2 = ac - 4 * dc;
//        int     no_edge = 0 <= r2 && r2 < rmax && 0 <= c2 && c2 < cmax;
        int     no_edge = gz_start_r <= r2 && r2 <= gz_end_r && gz_start_c <= c2 && c2 <= gz_end_c;

        int     count1 = 4;
        for (int i = 0; i < 4 && count1 == 4; ++i)  // Check 4 previous cells
        {
            r1 -= dr;
            c1 -= dc;
            if (no_edge)
            {
                if (board[p + r1 * cmax + c1] == 0)     // Stop if no stone found
                    count1 = i;                     // Save stone align length (Default at 4)
            }
//            else if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax || board[p + r1 * cmax + c1] == 0)
            else if (r1 < gz_start_r || gz_end_r < r1 || c1 < gz_start_c || gz_end_c < c1 || board[p + r1 * cmax + c1] == 0)
                count1 = i;
        }

        r1 = ar;
        c1 = ac;
        r2 = ar + 4 * dr;
        c2 = ac + 4 * dc;
        no_edge = gz_start_r <= r2 && r2 <= gz_end_r && gz_start_c <= c2 && c2 <= gz_end_c;
//        no_edge = 0 <= r2 && r2 < rmax && 0 <= c2 && c2 < cmax;

        int     count2 = 4;
        for (int i = 0; i < 4 && count2 == 4; ++i)
        {
            r1 += dr;
            c1 += dc;
            if (no_edge)
            {
                if (board[p + r1 * cmax + c1] == 0)
                    count2 = i;
            }
            else if (r1 < gz_start_r || gz_end_r < r1 || c1 < gz_start_c || gz_end_c < c1 || board[p + r1 * cmax + c1] == 0)
//            else if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax || board[p + r1 * cmax + c1] == 0)
                count2 = i;
        }

        if (count1 + count2 >= 4)       // +1 for current action in the middle
        {
            // printf("WIN HEREE %d %d\n", count1, count2);
            return 1;
        }
        // else printf("NO WIN %d %d\n", count1, count2);
    }
    return 0;
}


//int basic_rule_winning(char *board, int ar, int ac)
//{
//    static int  direction[8] = {-1, -1, -1, 0, -1, 1, 0, -1};
//    static int  rmax = 19, cmax = 19;
//
//    // four directions
//    for (int direction_idx = 0; direction_idx < 8; direction_idx += 2)
//    {
//        // slide 4 times
//        int     dr = direction[direction_idx];
//        int     dc = direction[direction_idx + 1];
//        int     r1 = ar;
//        int     c1 = ac;
//        int     count1 = 4;
//
//        for (int i = 0; i < 4 && count1 == 4; ++i)
//        {
//            r1 += dr;
//            c1 += dc;
//            if (r1 < 0 || r1 >= rmax || c1 < 0 || c1 >= cmax || board[r1 * cmax + c1] == 0)
//                count1 = i;
//        }
//
//        int     r2 = ar;
//        int     c2 = ac;
//        int     count2 = 4;
//
//        for (int i = 0; i < 4 && count2 == 4; ++i)
//        {
//            r2 -= dr;
//            c2 -= dc;
//            if (r2 < 0 || r2 >= rmax || c2 < 0 || c2 >= cmax || board[r2 * cmax + c2] == 0)
//                count2 = i;
//        }
//
//        if (count1 + count2 + 1 >= 5)
//        {
//            // printf("WIN HEREE %d %d\n", count1, count2);
//            return 1;
//        }
//        // else printf("NO WIN %d %d\n", count1, count2);
//    }
//    return 0;
//}