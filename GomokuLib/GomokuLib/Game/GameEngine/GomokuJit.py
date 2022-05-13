import numpy as np

from typing import Union
from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Game.Action.GomokuAction import GomokuAction
from GomokuLib.Game.Rules import GameEndingCaptureJit, NoDoubleThreesJit, CaptureJit, BasicRuleJit, RULES, ForceWinOpponent, ForceWinPlayer


class GomokuJit(Gomoku):

    capture_class: object = CaptureJit

    def init_rules_fn(self, rules: list[Union[str, object]]):

        tab_rules = {
            'capture': self.capture_class,
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
            # rule.get_valid(self.state.full_board)
            rule.get_valid(np.ascontiguousarray(self.state.board[0] | self.state.board[1]).astype(np.int8))
            for rule in self.rules_fn['restricting']
        ])
        masks = np.bitwise_and.reduce(masks, axis=0)
        return masks

    # def apply_action(self, action: GomokuAction) -> None:
    #     ar, ac = action.action
    #     if not self.is_valid_action(action):
    #         print(f"Not a fucking valid action: {ar} {ac}")
    #         breakpoint()
    #         raise Exception

    #     self.state.board[0, ar, ac] = 1
    #     self.update_game_zone(ar, ac)
    #     self.last_action = action

    def is_valid_action(self, action: GomokuAction) -> bool:
        return all(
            rule.is_valid(np.ascontiguousarray(self.state.board[0] | self.state.board[1]).astype(np.int8), *action.action)
            for rule in self.rules_fn['restricting']
        )

    def _next_turn_rules(self):
        gz = self.get_game_zone()

        for rule in self.rules_fn['endturn']:  # A mettre dans le apply_action ?
            rule.endturn(self.player_idx, *self.last_action.action, gz[0], gz[1], gz[2], gz[3])

        # print(self.rules_fn['winning'])

        win = False
        for rule in self.rules_fn['winning']:
            flag = rule.winning(self.player_idx, *self.last_action.action, gz[0], gz[1], gz[2], gz[3])
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
            rule.nowinning(self.player_idx, self.last_action.action)
            for rule in self.rules_fn['nowinning']
        ])):
            self._isover = True
            self.winner = self.player_idx  # ????????????????????????????? Mouais

    def next_turn(self, *args, **kwargs):
        ret = super().next_turn(*args, **kwargs)
        for rule in self.rules:
            rule.update_board_ptr(self.state.board)
        return ret

    def _update_rules(self, engine: Gomoku):
        for to_update, rule in zip(self.rules, engine.rules):
            to_update.update(rule)
            to_update.update_board_ptr(self.state.board)

    def clone(self) -> Gomoku:
        engine = GomokuJit(self.players, self.board_size, self.rules_str)
        engine.update(self)
        return engine