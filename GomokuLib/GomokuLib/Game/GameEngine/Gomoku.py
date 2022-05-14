
from __future__ import annotations
from typing import Union, TYPE_CHECKING
import numpy as np

from GomokuLib.Game.Rules import GameEndingCapture, NoDoubleThrees, Capture, BasicRule, RULES
from GomokuLib.Game.Rules import ForceWinOpponent, ForceWinPlayer

from ..State.GomokuState import GomokuState

if TYPE_CHECKING:
    from ..Rules.Capture import Capture
    from ...Player.AbstractPlayer import AbstractPlayer

from .AbstractGameEngine import AbstractGameEngine

class Gomoku(AbstractGameEngine):

    capture_class: object = Capture

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]] = None,
                 board_size: Union[int, tuple[int]] = 19,
                 rules: list[str] = ['Capture', 'Game-Ending Capture', 'no double-threes'],
                 **kwargs) -> None:
        super().__init__(players)

        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.rules_str = rules
        self.init_game()

    def init_game(self, **kwargs):
        self.state = self.init_board()
        self.C, self.H, self.W = self.state.board.shape
        self.turn = 0
        self.last_action = (-1, -1)
        self._isover = False
        self.winner = -1
        self.player_idx = 0
        self.rules_fn = self.init_rules_fn(self.rules_str.copy())
        self._search_capture_rule()
        self.history = []
        self.game_zone = np.array(([0, 0, self.board_size[0] - 1, self.board_size[1] - 1]), dtype=np.int8)

    def set_rules_fn(self):
        self.rules_fn = {
            k: [r for r in self.rules if (hasattr(r, k) and getattr(r, k) != None)]
            for k in RULES
        }

    def init_rules_fn(self, rules: list[Union[str, object]]):

        tab_rules = {
            'capture': self.capture_class,
            'game-ending capture': GameEndingCapture,
            'no double-threes': NoDoubleThrees
        }
        rules.append(BasicRule(self.state.board))

        self.rules = [
            tab_rules[r.lower()](self.state.board)  # Attention ! Si la str n'est pas dans tab !
            if isinstance(r, str)
            else r
            for r in rules
        ]

        self.set_rules_fn()
        return self.rules_fn

    def init_board(self):
        """
            np array but we should go bitboards next
        """
        return GomokuState(self.board_size)

    def get_actions(self) -> np.ndarray:

        masks = np.array([
            # rule.get_valid(self.state.full_board)
            rule.get_valid(np.ascontiguousarray(self.state.board[0] | self.state.board[1]).astype(np.int8))
            for rule in self.rules_fn['restricting']
        ])
        masks = np.bitwise_and.reduce(masks, axis=0)
        return masks


    def is_valid_action(self, action: tuple[int]) -> bool:
        return all(
            rule.is_valid(np.ascontiguousarray(self.state.board[0] | self.state.board[1]).astype(np.int8), *action)
            for rule in self.rules_fn['restricting']
        )

    def apply_action(self, action: tuple[int]) -> None:
        ar, ac = action
        if not self.is_valid_action(action):
            print(f"Not a fucking valid action: {ar} {ac}")
            breakpoint()
            raise Exception

        self.state.board[0, ar, ac] = 1
        self.update_game_zone(ar, ac)
        self.last_action = (ar, ac)

    def _search_capture_rule(self):
        for r in self.rules:
            if isinstance(r, self.capture_class):
                self.capture_rule = r
        if self.capture_rule:
            self.get_captures = self._get_captures_success
        else:
            self.get_captures = self._get_captures_failed

    def _get_captures_success(self) -> list:
        return self.capture_rule.get_current_player_captures(self.player_idx)

    def _get_captures_failed(self) -> list:
        return [0, 0]

    def get_captures(self) -> list:
        self._search_capture_rule()
        return self.get_captures()

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
        gz = self.get_game_zone()

        for rule in self.rules_fn['endturn']:  # A mettre dans le apply_action ?
            rule.endturn(self.player_idx, *self.last_action, gz[0], gz[1], gz[2], gz[3])

        # print(self.rules_fn['winning'])

        win = False
        for rule in self.rules_fn['winning']:
            flag = rule.winning(self.player_idx, *self.last_action, gz[0], gz[1], gz[2], gz[3])
            if flag == 3:   # GameEndingCapture win
                self._isover = True
                self.winner = self.player_idx ^ 1
                return
            if flag == 1:   # BasicRule win
                win = True
            elif flag == 2:   # Capture win
                self._isover = True
                self.winner = self.player_idx
                return

        if (win and not any([   #  Ca setr à rien !!!!!!!!!!!!!!!!!
            rule.nowinning(self.player_idx, self.last_action)
            for rule in self.rules_fn['nowinning']
        ])):
            self._isover = True
            self.winner = self.player_idx  # ????????????????????????????? Mouais

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

        for rule in self.rules:
            rule.update_board_ptr(self.state.board)
        return cb_return


    def isover(self):
        # print(f"Gomoku(): isover() return {self._isover}")
        return self._isover

    def _run(self, players: AbstractPlayer) -> AbstractPlayer:

        while not self.isover():
            player = players[self.player_idx]
            player_action = player.play_turn()

            self.apply_action(player_action)
            self.next_turn()

        print(f"Player {self.winner} win.")

    def run(self, players: list[AbstractPlayer]) -> AbstractPlayer:

        self.players = players
        for p in self.players:
            p.init_engine(self)

        self.init_game()
        self._run(self.players)
        return self.players[self.winner] if self.winner >= 0 else self.winner

    def create_snapshot(self):
        # for rule in self.rules:
        #     ss = rule.create_snapshot()
        #     if np.any(ss):
        #         for k, v in ss.items():
        #             print(k, v)
        #             print(type(k), type(v))
        #     print()
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
            'rules': {
                rule.name: rule.create_snapshot() for rule in self.rules
            }
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

        # if self.game_zone_init and snapshot['game_zone_init']:
        #     self.game_zone[:] = snapshot['game_zone']
        # elif snapshot['game_zone_init']:
        #     self.game_zone = snapshot['game_zone'].copy()
        # else:
        #     self.game_zone = None
        # self.game_zone_init = snapshot['game_zone_init']

        for rule in self.rules:
            rule.update_from_snapshot(snapshot['rules'][rule.name])

    def _update_rules(self, engine: Gomoku):
        for to_update, rule in zip(self.rules, engine.rules):
            to_update.update(rule)
            to_update.update_board_ptr(self.state.board)

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

        # if self.game_zone_init and engine.game_zone_init:
        #     self.game_zone[:] = engine.game_zone
        # elif engine.game_zone_init:
        #     self.game_zone = engine.game_zone.copy()
        # else:
        #     self.game_zone = None
        # self.game_zone_init = engine.game_zone_init

        self._update_rules(engine)
        self.set_rules_fn()

    def clone(self) -> Gomoku:
        engine = Gomoku(self.players, self.board_size, self.rules_str)
        engine.update(self)
        return engine