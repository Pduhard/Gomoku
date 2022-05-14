from numba.extending import models, register_model

from numba import types

class SnapshotType(types.Type):
    def __init__(self):
        super(SnapshotType, self).__init__(name='Snapshot')

@register_model(SnapshotType)
class IntervalModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('lo', types.float64),
            ('hi', types.float64),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)