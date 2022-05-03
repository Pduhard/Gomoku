#include "algo.h"

int     mcts_lazy_selection(float *policy, int *best_actions)
{
    int     i_end = 361;
    float   best_policy = -1;
    int     *best_action_start = best_actions;

    for (int i = 0; i < i_end; ++i, ++policy)
    {
        if (*policy >= best_policy)
        {
            if (*policy > best_policy)
            {
                best_policy = *policy;
                best_actions = best_action_start;
            }
            *best_actions++ = i / 19;
            *best_actions++ = i % 19;
        }
    }
    return (best_actions - best_action_start) / 2;
}

void    init_random_buffer(int *random_buffer, int size)
{
    for (int i = 0; i < size; ++i)
        random_buffer[i] = i;
    
    // shuffle size times
    int rnd = rand();
    int rnd_idx;
    if (size < 2)
        return;
    for (int i = size - 1; i > 0; --i)
    {
        rnd = (rnd << 13) ^ 7;
        rnd = (rnd >> 5) ^ (rnd << 8);
        rnd = (rnd >> 9) ^ (rnd << 3);
        rnd_idx = rnd % i;
        random_buffer[i] ^= random_buffer[rnd_idx];
        random_buffer[rnd_idx] ^= random_buffer[i];
        random_buffer[i] ^= random_buffer[rnd_idx];
    }
}