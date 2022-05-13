
from __future__ import annotations
from typing import Union, TYPE_CHECKING
import numpy as np

from GomokuLib.Game.Rules import GameEndingCapture, NoDoubleThrees, Capture, BasicRule, RULES
from GomokuLib.Game.Rules import ForceWinOpponent, ForceWinPlayer

from ..State.GomokuState import GomokuState

from ..Action.GomokuAction import GomokuAction
if TYPE_CHECKING:
    from ..Rules.Capture import Capture
    # from ..Action.GomokuAction import GomokuAction
    from ..Rules.AbstractRule import AbstractRule
    from ...Player.AbstractPlayer import AbstractPlayer

from .AbstractGameEngine import AbstractGameEngine


class Gomoku(AbstractGameEngine):

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]] = None,
                 board_size: Union[int, tuple[int]] = 19,
                 rules: list[Union[str, AbstractRule]] = ['Capture', 'Game-Ending Capture', 'no double-threes'],
                 **kwargs) -> None:
        super().__init__(players)

        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.rules_str = rules
        self.init_game()
        self.capture_rule = None

    def init_game(self, **kwargs):
        self.turn = 0
        self.last_action = None
        self._isover = False
        self.winner = -1
        self.player_idx = 0
        self.state = self.init_board()
        self.C, self.H, self.W = self.state.board.shape
        self.rules_fn = self.init_rules_fn(self.rules_str.copy())
        # self.history = np.zeros((1, self.C, self.H, self.W), dtype=int)
        self.history = []
        self.game_zone = [*self.board_size, 0, 0]

    def set_rules_fn(self, rules):
        self.rules_fn = {
            k: [r for r in rules if (hasattr(r, k) and getattr(r, k) != None)]
            for k in RULES
        }

    def init_rules_fn(self, rules: list[Union[str, AbstractRule]]):

        tab = {
            'capture': Capture,
            'game-ending capture': GameEndingCapture,
            'no double-threes': NoDoubleThrees
        }

        rules.append(BasicRule(self))
        self.rules = [
            tab[r.lower()](self)    # Attention ! Si la str n'est pas dans tab !
            if isinstance(r, str)
            else r
            for r in rules
        ]
        self.set_rules_fn(self.rules)
        return self.rules_fn

    def init_board(self):
        """
            np array but we should go bitboards next
        """
        return GomokuState(self.board_size)

    def get_actions(self) -> np.ndarray:

        # actions = self.rules_fn['restricting'][0].get_valid(self.state)
        masks = np.array([rule.get_valid() for rule in self.rules_fn['restricting']])
        masks = np.bitwise_and.reduce(masks, axis=0)
        # print(masks.shape)
        # print(masks)
        return masks

    def is_valid_action(self, action: GomokuAction) -> bool:
        isvalid = all(rule.is_valid(action) for rule in self.rules_fn['restricting'])
        return isvalid

    def apply_action(self, action: GomokuAction) -> None:
        ar, ac = action.action
        if not self.is_valid_action(action):
            print('action', action)
            breakpoint()
            print(f"Not a fucking valid action: {ar} {ac}")
            raise Exception
        # print(ar, ac)
        self.state.board[0, ar, ac] = 1
        self.update_game_zone(ar, ac)
        self.last_action = action

    def _get_captures_success(self):
        return self.capture_rule.get_current_player_captures()

    def _get_captures_failed(self):
        return [0, 0]

    def get_captures(self):
        for r in self.rules:
            if isinstance(r, Capture):
                self.capture_rule = r
        if self.capture_rule:
            self.get_captures = self._get_captures_success
        else:
            self.get_captures = self._get_captures_failed
        return self.get_captures()

    def get_history(self) -> np.ndarray:
        # return self.history[1:]
        return np.array(self.history)

    def update_game_zone(self, ar, ac):
        gz_start_r, gz_start_c, gz_end_r, gz_end_c = self.game_zone
        if ar < gz_start_r:
            self.game_zone[0] = ar
        elif gz_end_r < ar:
            self.game_zone[2] = ar
        if ac < gz_start_c:
            self.game_zone[1] = ac
        elif gz_end_c < ac:
            self.game_zone[3] = ac
        # if self.players:
        #     print(f"Game zone: {self.game_zone[0]} {self.game_zone[1]} into {self.game_zone[2]} {self.game_zone[3]}")

    def next_turn(self, before_next_turn_cb=[]) -> None:

        board = self.state.board if self.player_idx == 0 else self.state.board[::-1, ...]
        # self.history = np.insert(self.history, len(self.history), board, axis=0)
        # np.append(self.history, board[np.newaxis, ...], axis=0)
        # self.history = np.concatenate((self.history, board[np.newaxis, ...]))
        self.history.append(board)

        # print(f"GOMOKU NEXT TURN\n")

        # if np.all(self.state.full_board != 0):
        #     print("DRAW")
        #     self._isover = True
        #     self.winner = -1
        #     return
        try:

            if self.last_action is None:
                breakpoint()

            for rule in self.rules_fn['endturn']:       # A mettre dans le apply_action ?
                rule.endturn(self.last_action)

            # print(self.rules_fn['winning'])
            if (any([
                    rule.winning(self.last_action)
                    for rule in self.rules_fn['winning']
                ]) and not any([
                    rule.nowinning(self.last_action)
                    for rule in self.rules_fn['nowinning']
                ])):

                self._isover = True
                self.winner = self.player_idx       # ?????????????????????????????

        except ForceWinPlayer as e:
            # print(f"Player {self.player_idx} win. Reason: {e.reason}")
            self._isover = True
            self.winner = self.player_idx

        except ForceWinOpponent as e:
            # print(f"Player {self.player_idx ^ 1} win. Reason: {e.reason}")
            self._isover = True
            self.winner = self.player_idx ^ 1

        except Exception as e:
            print(f"An exception occur: {e}")
            exit(0)

        # print(f"GOMOKU CALLBACKS\n")
        cb_return = {}
        for cb in before_next_turn_cb:  # Callbacks
            cb_return.update(cb(self))

        self.turn += 1
        self.player_idx ^= 1
        self.state.board = self.state.board[::-1, ...]
        if not self.state.board.flags['C_CONTIGUOUS']:
            self.state.board = np.ascontiguousarray(self.state.board)

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
            print(f"Game zone: {self.game_zone[0]} {self.game_zone[1]} into {self.game_zone[2]} {self.game_zone[3]}")

        print(f"Player {self.winner} win.")

    def run(self, players: list[AbstractPlayer]) -> AbstractPlayer:

        self.players = players
        for p in self.players:
            p.init_engine(self)

        self.init_game()
        self._run(self.players)
        return self.players[self.winner] if self.winner >= 0 else self.winner


    def create_snapshot(self):
        return {
            'history': self.history.copy(),
            'last_action': None if self.last_action is None else GomokuAction(*self.last_action.action),
            'board': self.state.board.copy(),
            'player_idx': self.player_idx,
            '_isover': self._isover,
            'winner': self.winner,
            'turn': self.turn,
            'game_zone': self.game_zone,
            'rules': {
                rule.name: rule.create_snapshot() for rule in self.rules
            }
        }

    def update_from_snapshot(self, snapshot):
        self.history = snapshot['history'].copy()
        self.last_action = None if snapshot['last_action'] is None else GomokuAction(*snapshot['last_action'].action)
        self.state.board = snapshot['board'].copy()
        self.player_idx = snapshot['player_idx']
        self._isover = snapshot['_isover']
        self.winner = snapshot['winner']
        self.turn = snapshot['turn']
        self.game_zone[:] = snapshot['game_zone']
        for rule in self.rules:
            rule.update_from_snapshot(snapshot['rules'][rule.name])

    def update(self, engine: Gomoku):

        self.history = engine.history.copy()
        self.last_action = None if engine.last_action is None else GomokuAction(*engine.last_action.action)
        self.state.board = engine.state.board.copy()

        self.player_idx = engine.player_idx
        self._isover = engine._isover
        self.winner = engine.winner
        self.turn = engine.turn
        self.game_zone[:] = engine.game_zone

        # print(engine.rules_fn)
        # print(self.rules_fn)
        # print('----------------------')
        self.rules = [rule.copy(self, rule) for rule in engine.rules]
        self.set_rules_fn(self.rules)
        # print(engine.rules_fn)
        # print(self.rules_fn)

    def clone(self) -> Gomoku:
        engine = Gomoku(self.players, self.board_size, self.rules_str)
        engine.update(self)
        return engine