#include <time.h>
#include <stdlib.h>
#include <stdio.h>

void    init_random();
void    init_random_buffer(int *random_buffer, int size);

int     mcts_lazy_selection(double *policy, int *best_actions);

int     gettime();
