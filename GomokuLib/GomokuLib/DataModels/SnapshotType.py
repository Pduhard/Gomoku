from numba import types
from numba.core import cgutils
from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    lower_builtin,
    unbox,
    NativeValue,
)

class Snapshot(object):

    def __init__(self, history, last_action, board, player_idx,
        _isover, winner, turn, game_zone) -> None:
        self.history = history
        self.last_action = last_action
        self.board = board
        self.player_idx = player_idx
        self._isover = _isover
        self.winner = winner
        self.turn = turn
        self.game_zone = game_zone

class SnapshotType(types.Type):
    def __init__(self):
        super(SnapshotType, self).__init__(name='Snapshot')

@register_model(SnapshotType)
class Snapshot(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('history', types.CPointer(types.int8)),
            ('last_action', types.int16),
            ('board', types.CPointer(types.int8)),
            ('player_idx', types.types.int64),
            ('_isover', types.types.int64),
            ('winner', types.types.int64),
            ('turn', types.types.int64),
            ('game_zone', types.CPointer(types.int8)),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

make_attribute_wrapper(SnapshotType, 'history', 'history')
make_attribute_wrapper(SnapshotType, 'last_action', 'last_action')
make_attribute_wrapper(SnapshotType, 'board', 'board')
make_attribute_wrapper(SnapshotType, 'player_idx', 'player_idx')
make_attribute_wrapper(SnapshotType, '_isover', '_isover')
make_attribute_wrapper(SnapshotType, 'winner', 'winner')
make_attribute_wrapper(SnapshotType, 'turn', 'turn')
make_attribute_wrapper(SnapshotType, 'game_zone', 'game_zone')

@lower_builtin(
    Snapshot,
    types.CPointer(types.int8),
    types.int16,
    types.CPointer(types.int8),
    types.types.int64,
    types.types.int64,
    types.types.int64,
    types.types.int64,
    types.CPointer(types.int8))
def impl_snapshot(context, builder, sig, args):
    typ = sig.return_type
    history, last_action, board, player_idx, _isover, winner, turn, game_zone = args
    snapshot = cgutils.create_struct_proxy(typ)(context, builder)
    snapshot.history = history
    snapshot.last_action = last_action
    snapshot.board = board
    snapshot.player_idx = player_idx
    snapshot._isover = _isover
    snapshot.winner = winner
    snapshot.turn = turn
    snapshot.game_zone = game_zone
    return snapshot._getvalue()



@unbox(SnapshotType)
def unbox_interval(typ, obj, c):
    """
    Convert a Interval object to a native interval structure.
    """
    history_obj = c.pyapi.object_getattr_string(obj, "history")
    last_action_obj = c.pyapi.object_getattr_string(obj, "last_action")
    board_obj = c.pyapi.object_getattr_string(obj, "board")
    player_idx_obj = c.pyapi.object_getattr_string(obj, "player_idx")
    _isover_obj = c.pyapi.object_getattr_string(obj, "_isover")
    snapshot._isover = _isover
    snapshot.winner = winner
    snapshot.turn = turn
    snapshot.game_zone = game_zone
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    interval.lo = c.pyapi.float_as_double(history_obj)
    interval.hi = c.pyapi.float_as_double(last_action_obj)
    c.pyapi.decref(history_obj)
    c.pyapi.decref(last_action_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(interval._getvalue(), is_error=is_error)