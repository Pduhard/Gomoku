from sre_parse import State
from numba import types
from numba.core import cgutils
from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    lower_builtin,
    unbox,
    box,
    NativeValue,
    typeof_impl,
    type_callable,
    overload_attribute
)
import numpy as np

class StateData(object):

    def __init__(self, visit: int, reward: float, state_action: np.ndarray, action: np.ndarray) -> None:
        self.visit = visit
        self.reward = reward
        self.state_action = state_action
        self.action = action

class StateDataType(types.Type):
    def __init__(self):
        self.visit = types.int64
        self.reward = types.float32
        self.state_action = types.Array(dtype=types.float32, ndim=2, layout="C")
        self.action =  types.Array(dtype=types.int32, ndim=2, layout="C")
        super(StateDataType, self).__init__(name='StateData')


state_data_type = StateDataType()
@typeof_impl.register(StateData)
def typeof_state_data(val, c):
    return state_data_type


@type_callable(StateData)
def type_state_data(context):
    def typer(visit, reward, state_action, action):
        if (isinstance(visit, types.Integer)
            and isinstance(reward, types.Float)
            and isinstance(state_action,types.Array)
            and isinstance(action,types.Array)
        ):
            return state_data_type
    return typer

@overload_attribute(types.Array, 'nbytes')
def array_nbytes(arr):
   def get(arr):
       return arr.size * arr.itemsize
   return get


@register_model(StateDataType)
class StateDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('visit', fe_type.visit),
            ('reward', fe_type.reward),
            ('state_action', fe_type.state_action),
            ('action',  fe_type.action),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(StateDataType, 'visit', 'visit')
make_attribute_wrapper(StateDataType, 'reward', 'reward')
make_attribute_wrapper(StateDataType, 'state_action', 'state_action')
make_attribute_wrapper(StateDataType, 'action', 'action')

@lower_builtin(
    StateData,
    types.int64,
    types.float32,
    types.Array(dtype=types.float32, ndim=2, layout="C"),
    types.Array(dtype=types.int32, ndim=2, layout="C"),
)
def impl_state_data(context, builder, sig, args):
    typ = sig.return_type
    action, reward, state_action, action = args
    state_data = cgutils.create_struct_proxy(typ)(context, builder)
    state_data.action = action
    state_data.reward = reward
    state_data.state_action = state_action
    state_data.action = action
    return state_data._getvalue()

@unbox(StateDataType)
def unbox_state_data(typ, obj, c):
    """
    Convert a StateData object to a native State Data structure.
    """
    visit_obj = c.pyapi.object_getattr_string(obj, "visit")
    reward_obj = c.pyapi.object_getattr_string(obj, "reward")
    state_action_obj = c.pyapi.object_getattr_string(obj, "state_action")
    action_obj = c.pyapi.object_getattr_string(obj, "action")

    state_data = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    # state_data.visit = c.pyapi.from_long(visit_obj)
    # state_data.reward = c.pyapi.from_float(reward_obj)
   
    state_data.visit = c.unbox(typ.visit, visit_obj).value
    state_data.reward = c.unbox(typ.reward, reward_obj).value
    state_data.state_action = c.unbox(typ.state_action, state_action_obj).value
    state_data.action = c.unbox(typ.action, action_obj).value
    c.pyapi.decref(visit_obj)
    c.pyapi.decref(reward_obj)
    c.pyapi.decref(state_action_obj)
    c.pyapi.decref(action_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(state_data._getvalue(), is_error=is_error)

@box(StateDataType)
def box_state_data(typ, val, c):
    state_data = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    classobj = c.pyapi.unserialize(
            c.pyapi.serialize_object(state_data))
    visit_obj = c.box(typ.visit, state_data.visit)
    reward_obj = c.box(typ.reward, state_data.reward)
    state_action_obj = c.box(typ.state_action, state_data.state_action)
    action_obj = c.box(typ.action, state_data.action)
    state_data_obj = c.pyapi.call_function_objargs(classobj, (
        visit_obj, reward_obj, state_action_obj, action_obj))
    return state_data_obj

from numba import njit

@njit
def add(a: StateData, b: StateData):
    return StateData(
        a.visit + b.visit,
        a.reward + b.reward,
        a.state_action + b.state_action,
        a.action + b.action,
    )

if __name__ == "__main__":
    a = StateData(0, 0, np.ones((19, 19), dtype=np.float32), np.ones((19, 19), dtype=np.int32))

    b = StateData(4, 6, np.ones((19, 19), dtype=np.float32), np.ones((19, 19), dtype=np.int32))

    c = add(a, b)

    print(a.action), b.action, c.action