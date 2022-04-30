#include "mctsrules.h"

int capture(int a, int b)
{
    int c;
    for (int j = 0; j < 1000000; ++j)
        c = a + b;
    return c;
}

int npadd(float a, float *b)
{
    for (int j = 0; j < 1000000; ++j)
        for (int i = 0; i < 16; ++i)
            b[i] += a;
}