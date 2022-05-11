import numpy as np

from typing import Union
from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Game.Action.GomokuAction import GomokuAction
from GomokuLib.Game.Rules import GameEndingCaptureJit, NoDoubleThreesJit, CaptureJit, BasicRuleJit, RULES, ForceWinOpponent, ForceWinPlayer


class GomokuJit(Gomoku):

    def init_rules_fn(self, rules: list[Union[str, object]]):

        tab_rules = {
            'capture': CaptureJit,
            'game-ending capture': GameEndingCaptureJit,
            'no double-threes': NoDoubleThreesJit
        }
        rules.append(BasicRuleJit(self.state.board))

        self.rules = [
            tab_rules[r.lower()](self.state.board)  # Attention ! Si la str n'est pas dans tab !
            if isinstance(r, str)
            else r
            for r in rules
        ]

        self.set_rules_fn()
        return self.rules_fn

    def get_actions(self) -> np.ndarray:

        masks = np.array([
            rule.get_valid(self.state.full_board)
            for rule in self.rules_fn['restricting']
        ])
        masks = np.bitwise_and.reduce(masks, axis=0)
        return masks

    def is_valid_action(self, action: GomokuAction) -> bool:
        return all(
            rule.is_valid(self.state.full_board, *action.action)
            for rule in self.rules_fn['restricting']
        )

    def get_captures(self):
        return super().get_captures(capture_class=CaptureJit)

    def _next_turn_rules(self):
        gz = self.get_game_zone()

        for rule in self.rules_fn['endturn']:  # A mettre dans le apply_action ?
            rule.endturn(self.player_idx, *self.last_action.action, *gz)

        # print(self.rules_fn['winning'])
        if (any([
            rule.winning(self.player_idx, *self.last_action.action, *gz)
            for rule in self.rules_fn['winning']
        ]) and not any([
            rule.nowinning(self.player_idx, self.last_action.action)
            for rule in self.rules_fn['nowinning']
        ])):
            self._isover = True
            self.winner = self.player_idx  # ?????????????????????????????

    def _update_rules(self, engine: Gomoku):
        self.rules = [rule.copy(self.state.board) for rule in engine.rules]
        self.set_rules_fn(self.rules)
