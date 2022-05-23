
class Human:

    def __str__(self):
        return f"Human"

    def play_turn(self, runner) -> tuple[int]:
        return runner.wait_player_action()

