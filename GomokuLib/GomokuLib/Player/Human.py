
class Human:

    def __str__(self):
        return f"Human"

    def play_turn(self, engine) -> tuple[int]:
        return engine.wait_player_action()

