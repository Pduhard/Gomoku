import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from GomokuLib.Typing import StateDtype, nbState 

@njit
def add(a: StateDtype, b: StateDtype):
    # use classic creation array method like you use recarray in numba compiled code
    c = np.zeros(1, dtype=StateDtype)[0]
    po = a.visit + b.visit
    c.visit += a.visit + b.visit
    c.reward += a.reward + b.reward
    c.state_action += a.state_action + b.state_action
    c.action += a.action + b.action
    return c

@njit
def access(dt: nb.types.DictType , key_ref: types.unicode_type):
    for i in range(1000):
        key = str(i)
        dt[key] = add(dt[key_ref], dt[key_ref])


if __name__ == "__main__":

    # use recarray to create element in python interpretation mode
    a_rec = np.recarray(1, dtype=StateDtype)[0]
    dt = nb.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=nbState,
    )
    b_rec = np.recarray(1, dtype=StateDtype)[0]
    
    c_rec = add(a_rec, b_rec)
    key = "23"
    dt[key] = c_rec
    e = access(dt, key)

    from time import perf_counter

    s = perf_counter()
    for i in range(1000):
        access(dt, key)
    e = perf_counter()

    print(e - s)