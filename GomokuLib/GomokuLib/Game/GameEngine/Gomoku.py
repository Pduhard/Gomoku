
from __future__ import annotations
from typing import Union, TYPE_CHECKING
import numpy as np
import numba as nb
from numba import njit

from GomokuLib.Game.Rules import GameEndingCapture, NoDoubleThrees, Capture, BasicRule, RULES
from GomokuLib.Game.Rules import ForceWinOpponent, ForceWinPlayer

from ..State.GomokuState import GomokuState

if TYPE_CHECKING:
    from ..Rules.Capture import Capture

spec = [
	('name', nb.types.string),
	('board_size', nb.typeof((0, 0))),
	('is_capture_active', nb.types.boolean),
	('is_game_ending_capture_active', nb.types.boolean),
	('is_no_double_threes_active', nb.types.boolean),
	('_board_ptr', nb.types.CPointer(nb.types.int8)),
]

# @jitclass(spec)
class Gomoku:

    capture_class: object = Capture

    def __init__(self, board_size: Union[int, tuple[int]] = 19,
                 is_capture_active: bool = True,
                 is_game_ending_capture_active: bool = True,
                 is_no_double_threes_active: bool = True) -> None:
        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.is_capture_active = is_capture_active
        self.is_game_ending_capture_active = is_game_ending_capture_active
        self.is_no_double_threes_active = is_no_double_threes_active
        self.init_game()

    def init_game(self):
        self.state = self.init_board()
        self.C, self.H, self.W = self.state.board.shape
        self.turn = 0
        self.last_action = (-1, -1)
        self._isover = False
        self.winner = -1
        self.player_idx = 0
        self.history = []
        self.game_zone = np.array(([0, 0, self.board_size[0] - 1, self.board_size[1] - 1]), dtype=np.int8)
        self.capture = Capture(self.state.board)
        self.game_ending_capture = GameEndingCapture(self.state.board)
        self.no_double_threes = NoDoubleThrees(self.state.board)
        self.basic_rules = BasicRule(self.state.board)

    def init_board(self):
        """
            np array but we should go bitboards next
        """
        return GomokuState(self.board_size)

    def get_actions(self) -> np.ndarray:
        masks = np.ones((19, 19), dtype=np.int8)

        full_board = (self.state.board[0] | self.state.board[1]).astype(np.int8)
        masks |= self.basic_rules.get_valid(full_board)
        if self.no_double_threes:
            masks |= self.no_double_threes.get_valid(full_board)
        return masks


    def is_valid_action(self, action: tuple[int]) -> bool:
        ar, ac = action
        full_board = (self.state.board[0] | self.state.board[1]).astype(np.int8)
        is_valid = self.basic_rules.is_valid(full_board, ar, ac)
        if self.no_double_threes:
            is_valid |= self.no_double_threes.is_valid(full_board, ar, ac)
        return is_valid

    def apply_action(self, action: tuple[int]) -> None:
        ar, ac = action
        if not self.is_valid_action(action):
            print(f"Not a fucking valid action: {ar} {ac}")
            breakpoint()
            raise Exception

        self.state.board[0, ar, ac] = 1
        self.update_game_zone(ar, ac)
        self.last_action = (ar, ac)

    def get_captures(self) -> list:
        if self.is_capture_active:
            return self.capture.get_current_player_captures(self.player_idx)
        return [0, 0]

    def get_history(self) -> np.ndarray:
        return np.array(self.history)

    def get_game_zone(self) -> list:
        # return np.array(([0, 0, self.board_size[0] - 1, self.board_size[1] - 1]), dtype=np.int8)
        return self.game_zone

    def update_game_zone(self, ar, ac):
        if self.turn:
            if ar < self.game_zone[0]:
                self.game_zone[0] = ar
            elif self.game_zone[2] < ar:
                self.game_zone[2] = ar
            if ac < self.game_zone[1]:
                self.game_zone[1] = ac
            elif self.game_zone[3] < ac:
                self.game_zone[3] = ac
        else:
            # self.game_zone_init = True
            self.game_zone = np.array((ar, ac, ar, ac), dtype=np.int8)

    def _next_turn_rules(self):
        gz0, gz1, gz2, gz3 = self.get_game_zone()
        ar, ac = self.last_action
        if self.is_capture_active:
            self.capture.endturn(self.player_idx, ar, ac, gz0, gz1, gz2, gz3)
            if self.capture.winning(self.player_idx):
                self._isover = True
                self.winner = self.player_idx
                return

        if self.is_game_ending_capture_active:
            if self.game_ending_capture.winning(self.player_idx, ar, ac, gz0, gz1, gz2, gz3):
                self._isover = True
                self.winner = self.player_idx ^ 1
                return
            self.game_ending_capture.endturn(self.player_idx, ar, ac)
        elif self.basic_rules.winning(self.player_idx, ar, ac, gz0, gz1, gz2, gz3):
            self._isover = True
            self.winner = self.player_idx

    def next_turn(self, before_next_turn_cb=[]) -> None:

        board = self.state.board if self.player_idx == 0 else self.state.board[::-1, ...]
        self.history.append(board)

        # if np.all(self.state.full_board != 0):
        #     print("DRAW")
        #     self._isover = True
        #     self.winner = -1
        #     return

        if self.last_action[0] == -1:
            breakpoint()

        self._next_turn_rules()

        cb_return = {}
        for cb in before_next_turn_cb:  # Callbacks
            cb_return.update(cb(self))

        self.turn += 1
        self.player_idx ^= 1
        self.state.board = self.state.board[::-1, ...]
        if not self.state.board.flags['C_CONTIGUOUS']:
            self.state.board = np.ascontiguousarray(self.state.board)

        self.basic_rules.update_board_ptr(self.state.board)
        if self.is_capture_active:
            self.capture.update_board_ptr(self.state.board)
        if self.is_game_ending_capture_active:
            self.game_ending_capture.update_board_ptr(self.state.board)
        if self.is_no_double_threes_active:
            self.no_double_threes.update_board_ptr(self.state.board)
        return cb_return

    def isover(self):
        return self._isover

    def create_snapshot(self):
        ar, ac = self.last_action
        return {
            'history': self.history.copy(),
            # 'history': self.get_history(),
            'last_action': (ar, ac),
            'board': self.state.board.copy(),
            'player_idx': self.player_idx,
            '_isover': self._isover,
            'winner': self.winner,
            'turn': self.turn,
            'game_zone': self.get_game_zone(), # copy ???
            # 'game_zone_init': self.game_zone_init,
            'basic_rules_rules': self.basic_rules.create_snapshot(),
            'capture_rules': self.capture.create_snapshot(),
            'game_ending_capture_rules': self.game_ending_capture.create_snapshot(),
            'no_double_threes_rules': self.no_double_threes.create_snapshot(),
        }

    def update_from_snapshot(self, snapshot):
        ar, ac = snapshot['last_action']

        self.history = snapshot['history'].copy()
        self.last_action = (ar, ac)
        self.state.board = snapshot['board'].copy()
        self.player_idx = snapshot['player_idx']
        self._isover = snapshot['_isover']
        self.winner = snapshot['winner']
        self.turn = snapshot['turn']
        self.game_zone[:] = snapshot['game_zone']

        self.basic_rules.update_from_snapshot(snapshot['basic_rules_rules'])
        self.basic_rules.update_board_ptr(self.state.board)
        if self.is_capture_active:
            self.capture.update_from_snapshot(snapshot['capture_rules'])
            self.capture.update_board_ptr(self.state.board)
        if self.is_game_ending_capture_active:
            self.game_ending_capture.update_from_snapshot(snapshot['game_ending_capture_rules'])
            self.game_ending_capture.update_board_ptr(self.state.board)
        if self.is_no_double_threes_active:
            self.no_double_threes.update_from_snapshot(snapshot['no_double_threes_rules'])
            self.no_double_threes.update_board_ptr(self.state.board)

    def _update_rules(self, engine: Gomoku):
        self.basic_rules.update(engine.basic_rules)
        self.basic_rules.update_board_ptr(self.state.board)
        if self.is_capture_active:
            self.capture.update(engine.capture)
            self.capture.update_board_ptr(self.state.board)
        if self.is_game_ending_capture_active:
            self.game_ending_capture.update(engine.game_ending_capture)
            self.game_ending_capture.update_board_ptr(self.state.board)
        if self.is_no_double_threes_active:
            self.no_double_threes.update(engine.no_double_threes)
            self.no_double_threes.update_board_ptr(self.state.board)

    def update(self, engine: Gomoku):
        ar, ac = engine.last_action

        self.history = engine.history.copy()
        self.last_action = (ar, ac)
        self.state.board = engine.state.board.copy()

        self.player_idx = engine.player_idx
        self._isover = engine._isover
        self.winner = engine.winner
        self.turn = engine.turn
        self.game_zone[:] = engine.game_zone

        self._update_rules(engine)

    def clone(self) -> Gomoku:
        engine = Gomoku(self.board_size, self.is_capture_active,
            self.is_game_ending_capture_active, self.is_no_double_threes_active)
        engine.update(self)
        return engine