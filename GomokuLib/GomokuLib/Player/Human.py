from GomokuLib.Game.GameEngine.GomokuGUIRunner import GomokuGUIRunner


class Human:

    def __init__(self, runner: GomokuGUIRunner):
        if not hasattr(runner, "wait_player_action"):
            print(f"{self}: Runner past in constructor has no attribute wait_player_action.")
            exit()

    def __str__(self):
        return f"Human"

    def play_turn(self, runner) -> tuple[int]:
        return runner.wait_player_action()
