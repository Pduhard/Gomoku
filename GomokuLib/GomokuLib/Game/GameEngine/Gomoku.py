
from __future__ import annotations
import copy
from subprocess import call
from typing import Union, TYPE_CHECKING
import numpy as np

from GomokuLib.Game.Action.AbstractAction import AbstractAction
from pygame import key

from GomokuLib.Game.Rules import GameEndingCapture, NoDoubleThrees, Capture, BasicRule, RULES
from GomokuLib.Game.Rules import ForceWinOpponent, ForceWinPlayer

from ..Rules.Capture import Capture
from ..State.GomokuState import GomokuState

if TYPE_CHECKING:
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
        self.init_game()
        self.rules_fn = self.init_rules_fn(rules)


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
        self.isover = False
        self.player_idx = 0
        self.state = self.init_board()

    def init_board(self):
        """
            np array but we should go bitboards next
        """
        return GomokuState(self.board_size)
    #     return np.random.randint(-1, 2, self.board_size)



    def get_actions(self) -> list[GomokuAction]:

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

        self.state.board[0, ar, ac] = 1
        self.last_action = action


    def new_rule(self, obj: object, operation: str):
        # def rule(self, rule, action):
        #     cls, method = rule
        #     if cls is None:
        #         return method(action)
        #     return getattr(cls, method)(action)
        # pass

        self.rules_fn[operation].append(obj)
    
    def remove_rule(self, obj: object, operation: str):
        self.rules_fn[operation].remove(obj)


    def next_turn(self) -> None:
        for rule in self.rules_fn['endturn']:
            rule.endturn(self.last_action)

        print(self.rules_fn['winning'])
        self.isover = (
            any([rule.winning(self.last_action) for rule in self.rules_fn['winning']])
            and not any([rule.nowinning(self.last_action) for rule in self.rules_fn['nowinning']])
        )

        self.player_idx ^= 1
        self.state.board = self.state.board[::-1, ...]

    def isnotover(self):
        return self.isover is False

    def _run_turn(self, players: AbstractPlayer):
        player = players[self.player_idx]
        actions, state = self.get_actions(), self.state

        player_action = player.play_turn(state, actions)

        self.apply_action(player_action)
        self.next_turn()

    def _run(self, players: AbstractPlayer) -> AbstractPlayer:

        while self.isnotover():
            self._run_turn(players)

        print(f"Player {self.player_idx} win.")
        return self.player_idx ^ 1

    def run(self, players: list[AbstractPlayer]) -> AbstractPlayer:
        self.init_game()
        winner_idx = self._run(players)
        try:
            pass
        except ForceWinPlayer as e:
            print(f"Player {self.player_idx} win. Reason: {e.reason}")
            return players[self.player_idx]

        except ForceWinOpponent as e:
            print(f"Player {self.player_idx ^ 1} win. Reason: {e.reason}")
            return players[self.player_idx ^ 1]
        
        except Exception as e:
            print(f"An exception occur: {e}")
            exit(0)

        return players[winner_idx]


    def update(self, engine: Gomoku):

        print("Gomoku.copy()")
        self.state.board = engine.state.board.copy()
        self.isover = engine.isover
        self.player_idx = engine.player_idx
        self.rules_fn = {
            k : [rule.copy(self) for rule in rules]
            for k, rules in engine.rules_fn.items()
        }
        print(engine.rules_fn)
        print(self.rules_fn)

