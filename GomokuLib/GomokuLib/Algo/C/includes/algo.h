#include <time.h>
#include <stdlib.h>

int     mcts_lazy_selection(float *policy, int *best_actions);
void    init_random();
float   mcts_eval_heuristic(char *board)
