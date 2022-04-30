#include <time.h>
#include <stdlib.h>

int     mcts_lazy_selection(float *policy, int *best_actions);
void    init_random();
float   mcts_eval_heuristic(char *board);
void    init_random_buffer(int *random_buffer, int size);
