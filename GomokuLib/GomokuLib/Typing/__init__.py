import numba as nb
import numpy as np

from numba.core import types
### Size definition

BoardDtype = np.int8
ActionDtype = np.int8
GameZoneDtype = np.int8
PruningDtype = np.float32
HeuristicGraphDtype = np.float32

TupleDtype = np.int32
MCTSIntDtype = np.int32
MCTSFloatDtype = np.float32

# hpruning_dict = nb.typed.Dict.empty(
#     key_type=nb.int8,
#     value_type=nb.from_dtype(PruningDtype)
# )
# nbHpruningDict = nb.typeof(hpruning_dict)

StateDataDtype = np.dtype([
    # ('worker_id', MCTSIntDtype),
    ('max_depth', MCTSIntDtype),
    ('visits', MCTSIntDtype),
    ('rewards', MCTSFloatDtype),
    ('stateAction', MCTSFloatDtype, (2, 19, 19)),
    ('actions', ActionDtype, (19, 19)),
    ('heuristic', HeuristicGraphDtype),
    ('pruning', PruningDtype, (3, 19, 19)),
    ('amaf', MCTSFloatDtype, (2, 19, 19)),

    ('h_rewards', HeuristicGraphDtype, (21, 21)),
    ('h_captures', MCTSIntDtype, (2, ))
], align=True)


board_nb_dtype = nb.from_dtype(BoardDtype)
action_nb_type = nb.from_dtype(ActionDtype)
tuple_nb_dtype = nb.from_dtype(TupleDtype)
game_zone_nb_dtype = nb.from_dtype(GameZoneDtype)
heuristic_graph_nb_dtype = nb.from_dtype(HeuristicGraphDtype)
mcts_float_nb_dtype = nb.from_dtype(MCTSFloatDtype)
mcts_int_nb_dtype = nb.from_dtype(MCTSIntDtype)

state_data_nb_dtype = nb.from_dtype(StateDataDtype)

nbTuple = tuple_nb_dtype[:]
nbHeuristicGraph = nb.types.Array(dtype=heuristic_graph_nb_dtype, ndim=1, layout="C")
nbBoard = nb.types.Array(dtype=board_nb_dtype, ndim=3, layout="C")
nbAction = nb.types.Array(dtype=action_nb_type, ndim=2, layout="C")
nbByteArray = nb.types.Array(dtype=nb.uint8, ndim=1, layout="C")
nbGraph = nb.types.Array(dtype=nb.uint8, ndim=1, layout="C")

nbCapturedBuf = nb.types.Array(dtype=mcts_int_nb_dtype, ndim=3, layout="C")
nbByteArray = nb.types.Array(dtype=nb.uint8, ndim=1, layout="C")
nbPathArray = nb.types.Array(dtype=mcts_int_nb_dtype, ndim=2, layout="C")
nbHeuristicData = nb.types.Array(dtype=mcts_int_nb_dtype, ndim=2, layout="C")
nbBoardFFI = nb.types.CPointer(board_nb_dtype)
nbCapturedBufFFI = nb.types.CPointer(mcts_int_nb_dtype)

nbBoardFFI = nb.types.CPointer(board_nb_dtype)
nbGameZone = game_zone_nb_dtype[:]
nbPolicy = mcts_float_nb_dtype[:, :]
nbStrDtype = nb.typeof('en two one')

nbState = state_data_nb_dtype[:]
nbStateBuff = state_data_nb_dtype[:, :]


state_dict = nb.typed.Dict.empty(
    key_type=nb.types.unicode_type,
    ## obligé de mettre une recarray de taille 1 j'ai l'impression
    value_type=nbState,
)
nbStateDict = nb.typeof(state_dict)

heuristic_coefs_dict = nb.typed.Dict.empty(
    key_type=nb.types.unicode_type,
    value_type=heuristic_graph_nb_dtype
)
nbHeuristicCoefsDict = nb.typeof(heuristic_coefs_dict)


__all__ = [
    'TupleDtype',
    'nbTuple',

    'BoardDtype',
    'nbBoard',
    'nbBoardFFI',

    'ActionDtype',
    'nbAction',

    'PruningDtype',

    'nbState',

    'GameZoneDtype',
    'nbGameZone',
]