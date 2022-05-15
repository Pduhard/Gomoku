from struct import Struct
import numpy as np

import numba as nb
from numba import njit
from numba.core import types, cgutils
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy

from numba.extending import intrinsic
@structref.register
class StateDataType(types.StructRef):

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class StateData(structref.StructRefProxy):

    def __new__(cls, name, visit, reward, state_action, action):
        return structref.StructRefProxy.__new__(cls, name, visit, reward, state_action, action)

    @property
    def name(self):
        return StateData_get_name(self)


    @property
    def visit(self):
        return StateData_get_visit(self)


    @property
    def reward(self):
        return StateData_get_reward(self)

    @property
    def state_action(self):
        return StateData_get_state_action(self)

    @property
    def action(self):
        return StateData_get_action(self)


@njit
def StateData_get_name(self):
    return self.name

@njit
def StateData_get_visit(self):
    return self.visit

@njit
def StateData_get_reward(self):
    return self.reward

@njit
def StateData_get_state_action(self):
    return self.state_action

@njit
def StateData_get_action(self):
    return self.action

structref.define_proxy(StateData, StateDataType, ["name", "visit", "reward", "state_action", "action"])


@intrinsic
def _struct_from_meminfo(typingctx, struct_type, meminfo):
    inst_type = struct_type.instance_type

    def codegen(context, builder, signature, args):
        _, meminfo = args

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        st.meminfo = meminfo

        return st._getvalue()

    sig = inst_type(struct_type, types.MemInfoPointer(types.voidptr))
    return sig, codegen

@structref.register
class StateDataWrapperType(types.StructRef):

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class StateDataWrapper(structref.StructRefProxy):

    def __new__(cls, name, visit, state_data):
        return structref.StructRefProxy.__new__(cls, name, visit, state_data)

    @property
    def name(self):
        return StateDataWrapper_get_name(self)
    
    @property
    def visit(self):
        return StateDataWrapper_get_visit(self)

    @property
    def state_data(self):
        return StateDataWrapper_get_state_data(self)


@njit
def StateDataWrapper_get_visit(self):
    return self.visit

@njit
def StateDataWrapper_get_name(self):
    return self.name

@njit
def StateDataWrapper_get_state_data(self):
    return self.state_data

structref.define_proxy(StateDataWrapper, StateDataWrapperType, ["name", "visit", "state_data"])

@njit
def add(a: StateData, b: StateData):
    return StateData(
        "23", 
        a.visit + b.visit,
        a.reward + b.reward,
        a.state_action + b.state_action,
        a.action + b.action,
    )


@njit
def dataWrapper(a: StateData):
    return StateDataWrapper(
        "wrap0,",
        a.visit * 4,
        a,
    )

# erer = StateData("1", 0, 0, np.ones((19, 19), dtype=np.float32), np.ones((19, 19), dtype=np.int32))
@njit
def access(dt, key):
    meminfo = dt[key]
    vl = _struct_from_meminfo(StateDataType, meminfo)
    return vl.state_action,

if __name__ == "__main__":
    # dt = create_statedata_dict()
    dt = nb.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.MemInfoPointer(types.voidptr),
    )
    a = StateData("1", 0, 0, np.ones((19, 19), dtype=np.float32), np.ones((19, 19), dtype=np.int32))
    b = StateData("2", 4, 6, np.ones((19, 19), dtype=np.float32), np.ones((19, 19), dtype=np.int32))

    c = add(a, b)
    d = dataWrapper(c)
    key = "23"
    dt[key] = c._meminfo
    # for i in range()
    print(a.action, b.action, c.action, d.state_data.action, d.visit)

    e = access(dt, key)

# https://numba.discourse.group/t/any-numba-equivalent-for-casting-a-raw-pointer-to-a-structref-dict-list-etc/351/6
# https://numba.discourse.group/t/any-numba-equivalent-for-casting-a-raw-pointer-to-a-structref-dict-list-etc/351/6
# https://numba.discourse.group/t/any-numba-equivalent-for-casting-a-raw-pointer-to-a-structref-dict-list-etc/351/6