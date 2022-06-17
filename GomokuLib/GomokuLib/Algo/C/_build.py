import cffi
import os

PATH = os.path.dirname(__file__)

ffibuilder = cffi.FFI()
with open(os.path.join(PATH, 'srcs/time.c'), 'r') as f:

    ffibuilder.set_source(
        "fastcore._algo",
        f.read(),
        libraries=['c'],
        sources=[],
        include_dirs=[
            os.path.join(PATH, 'includes')
        ]
    )


ffibuilder.cdef(
    """
        int     gettime();
    """
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)