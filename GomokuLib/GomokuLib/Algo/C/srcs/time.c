#include "algo.h"

int     gettime()
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((int)tv.tv_sec) * 1000) + (int)(tv.tv_usec / 1000);
}
