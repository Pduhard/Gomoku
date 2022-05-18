from ast import Tuple
import numba as nb
import numpy as np

### Size definition

BoardDtype = np.int8
ActionDtype = np.int8
TupleDtype = np.int32
GameZoneDtype = np.int8
PathDtype = np.dtype([
    ('board', np.int8, (2, 19, 19)),
    ('player_idx', np.int32),
    ('bestaction', np.int32, (2,)),
], align=True)

StateDataDtype = np.dtype([
    ('Worker_id', np.int32),
    ('Depth', np.int32),
    ('Visits', np.int32),
    ('Rewards', np.float32),
    ('StateAction', np.float32, (2, 19, 19)),
    ('Actions', np.int32, (19, 19)),
    ('Heuristic', np.float32),
    # ('AMAF', np.int32, (2, 19, 19)),
], align=True)

board_nb_dtype = nb.from_dtype(BoardDtype)
action_nb_type = nb.from_dtype(ActionDtype)
tuple_nb_dtype = nb.from_dtype(TupleDtype)
game_zone_nb_dtype = nb.from_dtype(GameZoneDtype)
path_nb_dtype = nb.from_dtype(PathDtype)
path_array_nb_dtype = nb.typeof(np.zeros((), dtype=PathDtype))
state_data_nb_dtype = nb.from_dtype(StateDataDtype)
state_data_array_nb_dtype = nb.typeof(np.zeros((), dtype=StateDataDtype))

nbTuple = tuple_nb_dtype[:]
nbBoard = nb.types.Array(dtype=board_nb_dtype, ndim=3, layout="C")
nbAction = nb.types.Array(dtype=board_nb_dtype, ndim=2, layout="C")
nbBoardFFI = nb.types.CPointer(board_nb_dtype)
nbGameZone = game_zone_nb_dtype[:]
nbPath = path_nb_dtype[:]
nbPathArray = path_array_nb_dtype
nbStates = state_data_nb_dtype[:]
nbStateArray = state_data_array_nb_dtype



__all__ = [
    'TupleDtype',
    'nbTuple',

    'BoardDtype',
    'nbBoard',
    'nbBoardFFI',

    'ActionDtype',
    'nbAction',

    'GameZoneDtype',
    'nbGameZone',
]