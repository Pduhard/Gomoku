from numba import njit
import numpy as np

struct_dt = np.dtype([('nparr', np.ndarray), ('float_v', np.float64)])
struct_fname = ('nparr', 'float_v')

struct = np.zeros(4, dtype=struct_dt)

@njit()
def struct_mult(struct):

    for i in range(struct.)


struct_mult(struct)