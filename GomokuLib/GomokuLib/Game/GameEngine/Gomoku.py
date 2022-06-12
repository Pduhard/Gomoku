
from __future__ import annotations
from typing import Union, TYPE_CHECKING
import numpy as np
import numba as nb
from numba import njit

from numba.experimental import jitclass

from GomokuLib.Game.Rules import GameEndingCapture, NoDoubleThrees, Capture, BasicRule
import GomokuLib.Typing as Typing


@jitclass()
class Gomoku:

    ## TYPING
    board_size: Typing.nbTuple
    is_capture_active: nb.types.boolean
    is_game_ending_capture_active: nb.types.boolean
    is_no_double_threes_active: nb.types.boolean
    board: Typing.nbBoard
    turn: nb.types.int8
    last_action: Typing.nbTuple
    _isover: nb.types.boolean
    winner: nb.types.int32
    player_idx: nb.types.int32
    # history: nb.types.ListType(board_dtype)
    game_zone: Typing.nbGameZone
    capture: Capture
    game_ending_capture: GameEndingCapture
    no_double_threes: NoDoubleThrees
    basic_rules: BasicRule
    
    ##########

    def __init__(self, is_capture_active: bool = True,
                 is_game_ending_capture_active: bool = True,
                 is_no_double_threes_active: bool = True) -> None:
        self.board_size = np.array([19, 19], dtype=Typing.TupleDtype)
        self.is_capture_active = is_capture_active
        self.is_game_ending_capture_active = is_game_ending_capture_active
        self.is_no_double_threes_active = is_no_double_threes_active
        self.init_game()

    def init_game(self):
        self.board = np.zeros(shape=(2, 19, 19), dtype=Typing.BoardDtype)
        self.turn = 0
        self.last_action = np.array([-1, -1], dtype=Typing.TupleDtype)
        self._isover = False
        self.winner = -1
        self.player_idx = 0
        # self.history = []
        self.game_zone = np.array(
            [0, 0, self.board_size[0] - 1, self.board_size[1] - 1],
            dtype=Typing.GameZoneDtype)
        self.capture = Capture(self.board)
        self.game_ending_capture = GameEndingCapture(self.board)
        self.no_double_threes = NoDoubleThrees(self.board)
        self.basic_rules = BasicRule(self.board)

    def get_lazy_actions(self) -> np.ndarray:
        full_board = self.board[0] | self.board[1]
        return self.basic_rules.get_valid(full_board)

    def get_actions(self) -> np.ndarray:
        masks = np.ones((19, 19), dtype=Typing.BoardDtype)

        full_board = self.board[0] | self.board[1]
        masks &= self.basic_rules.get_valid(full_board)
        if self.is_no_double_threes_active:
            masks &= self.no_double_threes.get_valid(full_board, self.player_idx)
        return masks

    def is_valid_action(self, action: np.ndarray) -> bool:
        ar, ac = action
        full_board = self.board[0] | self.board[1]
        is_valid = self.basic_rules.is_valid(full_board, ar, ac)
        if self.is_no_double_threes_active:
            is_valid &= self.no_double_threes.is_valid(full_board, ar, ac, self.player_idx)
        return is_valid

    def apply_action(self, action: np.ndarray):
        ar, ac = action

        # if not self.is_valid_action(action):
        #     print("Not a fucking valid action:")
            # exit(0)
        #     breakpoint()
        #     raise Exception

        self.board[self.player_idx, ar, ac] = 1
        self.update_game_zone(ar, ac)
        self.last_action[0] = action[0]
        self.last_action[1] = action[1]

    def get_captures(self) -> list:
        if self.is_capture_active:
            return self.capture.get_captures()
        return np.array([0, 0], dtype=Typing.TupleDtype)

    def get_game_zone(self) -> list:
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
            self.game_zone = np.array((ar, ac, ar, ac), dtype=Typing.GameZoneDtype)

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
        return

    def _shift_board(self):

        self.turn += 1
        self.player_idx ^= 1
    
    def update_board_ptr(self):
        self.basic_rules.update_board_ptr(self.board)
        if self.is_capture_active:
            self.capture.update_board_ptr(self.board)
        if self.is_game_ending_capture_active:
            self.game_ending_capture.update_board_ptr(self.board)
        if self.is_no_double_threes_active:
            self.no_double_threes.update_board_ptr(self.board)

    def next_turn(self):
        self._next_turn_rules()
        self._shift_board()

    def isover(self):
        return self._isover

    def _update_rules(self, engine: Gomoku):
        self.basic_rules.update(engine.basic_rules)
        if self.is_capture_active:
            self.capture.update(engine.capture)
        if self.is_game_ending_capture_active:
            self.game_ending_capture.update(engine.game_ending_capture)
        if self.is_no_double_threes_active:
            self.no_double_threes.update(engine.no_double_threes)

    def update(self, engine: Gomoku):
        self.last_action = np.copy(engine.last_action)
        self.board = np.copy(engine.board)
        self.update_board_ptr()
        self.player_idx = engine.player_idx
        self._isover = engine._isover
        self.winner = engine.winner
        self.turn = engine.turn
        self.game_zone = np.copy(engine.game_zone)

        self._update_rules(engine)

    def clone(self) -> Gomoku:
        engine = Gomoku(self.is_capture_active,
            self.is_game_ending_capture_active, self.is_no_double_threes_active)
        engine.update(self)
        return engine