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