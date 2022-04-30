#include "algo.h"

int     mcts_lazy_selection(float *policy, int *best_actions)
{
    int     i_end = 19 * 19;
    float   best_policy = -1;
    int     best_action_count = 0;

    for (int i = 0; i < i_end; ++i)
    {
        if (policy[i] >= best_policy)
        {
            if (policy[i] > best_policy)
            {
                best_policy = policy[i];
                best_action_count = 0;
            }
            best_actions[best_action_count * 2] = i / 19;
            best_actions[best_action_count * 2 + 1] = i % 19;
            best_action_count++;
        }
    }
    return best_action_count;
}

void    init_random_buffer(int *random_buffer, int size)
{
    for (int i = 0; i < size; ++i)
        random_buffer[i] = i;
    
    // shuffle size times
    int a, b;
    if (size < 2)
        return;
    for (int i = 0; i < size; ++i)
    {
        a = rand() % size;
        b = rand() % size;
        while (b == a) 
            b = rand() % size;
        random_buffer[a] ^= random_buffer[b];
        random_buffer[b] ^= random_buffer[a];
        random_buffer[a] ^= random_buffer[b];
    }
}