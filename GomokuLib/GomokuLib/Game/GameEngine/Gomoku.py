
from __future__ import annotations
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
                 rules: list[Union[str, AbstractRule]] = ['Capture', 'Game-Ending Capture', 'no double-threes'],
                 board_size: Union[int, tuple[int]] = 19, **kwargs) -> None:
        super().__init__(players)
        print('init gomoku: ')
        for p in players:
            print(p.engine)
        self.players = players
        self.board_size = (board_size, board_size) if type(board_size) == int else board_size
        self.init_game()
        self.rules_fn = self.init_rules_fn(rules)
        self.callbacks_fn = {
            k: [] for k in RULES
        }

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
        print(rules_fn  )
        return rules_fn

    def init_board(self):
        """
            np array but we should go bitboards next
        """
        return GomokuState(self.board_size)
    #     return np.random.randint(-1, 2, self.board_size)

    def init_game(self):
        # init vars
        self.isover = False
        self.player_idx = 0
        self.current_player = self.players[0]
        # init board
        self.state = self.init_board()


    def set_state(self, GomokuState) -> None:
        pass

    def get_state(self) -> GomokuState:
        pass

    def get_actions(self) -> list[GomokuAction]:
        # for mask in [rule.get_valid(self.state) for rule in self.rules_fn['restricting']]:
        #     print(mask)
        
        pass

    def is_valid_action(self, action: GomokuAction) -> bool:

        # isvalid_cb = all(cb() for cb in self.callbacks_fn['restricting'])

        isvalid = all(rule.is_valid(action) for rule in self.rules_fn['restricting'])
        return isvalid

    def apply_action(self, action: GomokuAction) -> None:
        ar, ac = action.action

        # self.state.board[self.player_idx, ar, ac] = 1   # No Reverse Version
        self.state.board[0, ar, ac] = 1                   # Reverse Version
        # print(self.state.board)
        self.last_action = action

    def new_rule(self, obj: object, operation: str):
        self.rules_fn[operation].append(obj)
    
    def remove_rule(self, obj: object, operation: str):
        self.rules_fn[operation].remove(obj)

    # def rule(self, rule, action):
    #     cls, method = rule
    #     if cls is None:
    #         return method(action)
    #     return getattr(cls, method)(action)
        # pass

    def next_turn(self) -> None:
        for rule in self.rules_fn['endturn']:
            rule.endturn(self.last_action)

        print(self.rules_fn['winning'])

        # for rule in self.rules_fn['winning']:
        #     print(rule.winning)
        #     print(rule.winning(self.last_action))

        self.isover = (
            any([rule.winning(self.last_action) for rule in self.rules_fn['winning']])
            and not any([rule.nowinning(self.last_action) for rule in self.rules_fn['nowinning']])
        )
        print(self.rules_fn['winning'])
        print("-------")

        self.player_idx ^= 1
        self.current_player = self.players[self.player_idx]
        self.state.board = self.state.board[::-1, ...]

    def _run(self) -> AbstractPlayer:

            # game loop
            while self.isover is False:
                actions, state = self.get_actions(), self.get_state()
                player_action = self.current_player.play_turn(actions, state)
                if player_action not in actions:
                    print("player_action not in actions !")
                    exit(0)
                self.apply_action(player_action)
                self.next_turn()
            
            print(f"Player {self.player_idx} win.")
            return self.players[self.player_idx ^ 1]


    def run(self) -> AbstractPlayer:
        try:
            self._run()

        except ForceWinPlayer as e:
            print(f"Player {self.player_idx} win. Reason: {e.reason}")
            return self.players[self.player_idx]

        except ForceWinOpponent as e:
            print(f"Player {self.player_idx ^ 1} win. Reason: {e.reason}")
            return self.players[self.player_idx ^ 1]
        
        except Exception as e:
            print(f"An exception occur: {e}")
            exit(0)



