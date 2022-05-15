import numpy as np

from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy

@structref.register
class StateDataType(types.StructRef):

    def preprocess_fields(self, fields):
        return tuple((visit, types.unliteral(typ)) for visit, typ in fields)

class StateData(structref.StructRefProxy):

    def __new__(cls, visit, reward, state_action, action):
        return structref.StructRefProxy.__new__(cls, visit, reward, state_action, action)

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

structref.define_proxy(StateData, StateDataType, ["visit", "reward", "state_action", "action"])


@structref.register
class StateDataWrapperType(types.StructRef):

    def preprocess_fields(self, fields):
        return tuple((visit, types.unliteral(typ)) for visit, typ in fields)

class StateDataWrapper(structref.StructRefProxy):

    def __new__(cls, visit, state_data):
        return structref.StructRefProxy.__new__(cls, visit, state_data)

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
def StateDataWrapper_get_state_data(self):
    return self.state_data

structref.define_proxy(StateDataWrapper, StateDataWrapperType, ["visit", "state_data"])

@njit
def add(a: StateData, b: StateData):
    return StateData(
        a.visit + b.visit,
        a.reward + b.reward,
        a.state_action + b.state_action,
        a.action + b.action,
    )


@njit
def dataWrapper(a: StateData):
    return StateDataWrapper(
        a.visit * 4,
        a,
    )

if __name__ == "__main__":
    a = StateData(0, 0, np.ones((19, 19), dtype=np.float32), np.ones((19, 19), dtype=np.int32))
    b = StateData(4, 6, np.ones((19, 19), dtype=np.float32), np.ones((19, 19), dtype=np.int32))

    c = add(a, b)
    d = dataWrapper(c)

    print(a.action, b.action, c.action, d.state_data.action, d.visit)
