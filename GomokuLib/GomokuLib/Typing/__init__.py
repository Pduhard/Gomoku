from ast import Tuple
import numba as nb
import numpy as np

### Size definition

BoardDtype = np.int8
ActionDtype = np.int8
TupleDtype = np.int32
GameZoneDtype = np.int8
StateDtype = np.dtype([
    ('visit', 'i4'),
    ('reward', 'f4'),
    ('state_action', 'i4', (19, 19)),
    ('action', 'f4', (19, 19)),
], align=True)



board_nb_dtype = nb.from_dtype(BoardDtype)
action_nb_type = nb.from_dtype(ActionDtype)
tuple_nb_dtype = nb.from_dtype(TupleDtype)
game_zone_nb_dtype = nb.from_dtype(GameZoneDtype)
state_data_nb_type = nb.from_dtype(StateDtype)

nbTuple = tuple_nb_dtype[:]
nbBoard = nb.types.Array(dtype=board_nb_dtype, ndim=3, layout="C")
nbAction = nb.types.Array(dtype=board_nb_dtype, ndim=2, layout="C")
nbBoardFFI = nb.types.CPointer(board_nb_dtype)
nbGameZone = game_zone_nb_dtype[:]
nbState = state_data_nb_type


__all__ = [
    'TupleDtype',
    'nbTuple',

    'BoardDtype',
    'nbBoard',
    'nbBoardFFI',

    'ActionDtype',
    'nbAction',

    'StateDtype',
    'nbState',

    'GameZoneDtype',
    'nbGameZone',
]