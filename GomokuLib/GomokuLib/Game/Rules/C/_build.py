import cffi
import os

PATH = os.path.dirname(__file__)

ffibuilder = cffi.FFI()
with open(os.path.join(PATH, 'srcs/rules.c'), 'r') as f:

    ffibuilder.set_source(
        "fastcore._rules",
        f.read(),
        libraries=['c'],
        sources=[
            os.path.join(PATH, 'srcs/basic_rules.c'),
            os.path.join(PATH, 'srcs/capture.c'),
        ],
        include_dirs=[os.path.join(PATH, 'includes')])
    

with open(os.path.join(PATH, 'includes/rules.h'), 'r') as f:
    ffibuilder.cdef(f.read())

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)