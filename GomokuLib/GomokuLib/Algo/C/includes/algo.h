#include <time.h>
#include <stdlib.h>
#include <stdio.h>

void    init_random();
void    init_random_buffer(int *random_buffer, int size);

int     mcts_lazy_selection(float *policy, int *best_actions);
float   mcts_eval_heuristic(char *board, int cap_1, int cap_2);
