import cffi
import os

PATH = os.path.dirname(__file__)

ffibuilder = cffi.FFI()
with open(os.path.join(PATH, 'srcs/algo.c'), 'r') as f:

    ffibuilder.set_source(
        "fastcore._algo",
        f.read(),
        libraries=['c'],
        sources=[
            os.path.join(PATH, 'srcs/mcts_lazy.c'),
            os.path.join(PATH, 'srcs/mcts_eval.c'),
        ],
        include_dirs=[os.path.join(PATH, 'includes')])
    

ffibuilder.cdef(
    """
        void    init_random();
        void    init_random_buffer(int *random_buffer, int size);
        int     mcts_lazy_selection(double *policy, int *best_actions);
        float   mcts_eval_heuristic(char *board, int cap_1, int cap_2);
    """
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)