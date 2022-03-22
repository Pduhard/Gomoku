
from __future__ import annotations
from typing import Union, TYPE_CHECKING
import numpy as np

from GomokuLib.Game.Rules import GameEndingCapture, NoDoubleThrees, Capture, BasicRule, RULES
from GomokuLib.Game.Rules import ForceWinOpponent, ForceWinPlayer

from ..State.GomokuState import GomokuState

if TYPE_CHECKING:
    from ..Rules.Capture import Capture
    from ..Action.GomokuAction import GomokuAction
    from ..Rules.AbstractRule import AbstractRule
    from ...Player.AbstractPlayer import AbstractPlayer

from .AbstractGameEngine import AbstractGameEngine


class Gomoku(AbstractGameEngine):

    def __init__(self, players: Union[list[AbstractPlayer], tuple[AbstractPlayer]],
                 board_size: Union[int, tuple[int]] = 19,
                 rules: list[Union[str, AbstractRule]] = ['Capture', 'Game-Ending Capture', 'no double-threes'],
                 **kwargs) -> None:
        super().__init__(players)

        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.rules_fn = self.init_rules_fn(rules.copy())
        self.init_game()

    def init_rules_fn(self, rules: list[Union[str, AbstractRule]]):

        tab = {
            'capture': Capture,
            'game-ending capture': GameEndingCapture,
            'no double-threes': NoDoubleThrees
        }

        rules.append(BasicRule(self))
        rules = [tab[r.lower()](self) if isinstance(r, str) else r for r in rules]
        rules_fn = {
            k: [r for r in rules if (hasattr(r, k) and getattr(r, k) != None)]
            for k in RULES
        }
        return rules_fn

    def init_game(self):
        self.last_action = None
        self._isover = False
        self.winner = -1
        self.player_idx = 0
        self.state = self.init_board()
        self.C, self.H, self.W = self.state.board.shape
        self.history = np.zeros((1, self.C, self.H, self.W), dtype=int)

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
            breakpoint()
            print(f"Not a fucking valid action: {ar} {ac}")
            raise Exception
        # print(ar, ac)
        self.state.board[0, ar, ac] = 1
        self.last_action = action


    def new_rule(self, obj: object, operation: str):
        self.rules_fn[operation].append(obj)
    
    def remove_rule(self, obj: object, operation: str):
        self.rules_fn[operation].remove(obj)


    def get_history(self) -> np.ndarray:
        return self.history[1:]


    def next_turn(self) -> None:

        board = self.state.board if self.player_idx == 0 else self.state.board[::-1, ...]
        self.history = np.insert(self.history, len(self.history), board, axis=0)

        if np.all(self.state.full_board != 0):
            print("DRAW")
            self._isover = True
            self.winner = -1
            return
        try:

            for rule in self.rules_fn['endturn']:
                rule.endturn(self.last_action)

            # print(self.rules_fn['winning'])
            if (any([rule.winning(self.last_action) for rule in self.rules_fn['winning']])
                and not any([rule.nowinning(self.last_action) for rule in self.rules_fn['nowinning']])):

                self._isover = True
                self.winner = self.player_idx       # ?????????????????????????????

        except ForceWinPlayer as e:
            print(f"Player {self.player_idx} win. Reason: {e.reason}")
            self._isover = True
            self.winner = self.player_idx

        except ForceWinOpponent as e:
            print(f"Player {self.player_idx ^ 1} win. Reason: {e.reason}")
            self._isover = True
            self.winner = self.player_idx ^ 1

        except Exception as e:
            print(f"An exception occur: {e}")
            exit(0)

        finally:
            self.player_idx ^= 1
            self.state.board = self.state.board[::-1, ...]


    def isover(self):
        # print(f"Gomoku(): isover() return {self._isover}")
        return self._isover

    def _run_turn(self, players: AbstractPlayer):
        player = players[self.player_idx]

        player_action = player.play_turn()

        self.apply_action(player_action)
        self.next_turn()

    def _run(self, players: AbstractPlayer) -> AbstractPlayer:

        while not self.isover():
            self._run_turn(players)

        print(f"Player {self.winner} win.")

    def run(self, players: list[AbstractPlayer]) -> AbstractPlayer:
        self.init_game()
        for p in players:
            p.init_engine(self)

        self._run(players)

        return players[self.winner] if self.winner >= 0 else self.winner


    def update(self, engine: Gomoku):

        self.state.board = engine.state.board.copy()
        self.history = engine.history.copy()

        self._isover = engine._isover
        self.player_idx = engine.player_idx

        # print(engine.rules_fn)
        # print(self.rules_fn)
        # print('----------------------')
        self.rules_fn = {
            k : [rule.copy(self, rule) for rule in rules]
            for k, rules in engine.rules_fn.items()
        }
        # print(engine.rules_fn)
        # print(self.rules_fn)

