from cffi import FFI
import numpy as np
ffi = FFI()
class Snapshot:

    @staticmethod
    def create_snapshot(engine):
        return {
            'last_action': np.copy(engine.last_action),
            'board': np.copy(engine.board),
            'player_idx': engine.player_idx,
            'isover': engine._isover,
            'winner': engine.winner,
            'turn': engine.turn,
            'game_zone': np.copy(engine.get_game_zone()), # copy ???
            # 'game_zone_init': engine.game_zone_init,
            'basic_rules_rules': engine.basic_rules.create_snapshot(),
            'capture_rules': engine.capture.create_snapshot(),
            'game_ending_capture_rules': engine.game_ending_capture.create_snapshot(),
            'no_double_threes_rules': engine.no_double_threes.create_snapshot(),
        }

    @staticmethod
    def update_from_snapshot(engine, snapshot):
        engine.last_action = np.copy(snapshot['last_action'])
        engine.board = np.copy(snapshot['board'])

        engine.player_idx = snapshot['player_idx']
        engine._isover = snapshot['isover']
        engine.winner = snapshot['winner']
        engine.turn = snapshot['turn']
        engine.game_zone = np.copy(snapshot['game_zone'])

        engine.update_board_ptr()

        engine.basic_rules.update_from_snapshot(snapshot['basic_rules_rules'])
        if engine.is_capture_active:
            engine.capture.update_from_snapshot(snapshot['capture_rules'])
        if engine.is_game_ending_capture_active:
            engine.game_ending_capture.update_from_snapshot(snapshot['game_ending_capture_rules'])
        if engine.is_no_double_threes_active:
            engine.no_double_threes.update_from_snapshot(snapshot['no_double_threes_rules'])

