
from time import perf_counter

from GomokuLib.Game.GameEngine.Gomoku import Gomoku
from GomokuLib.Game.GameEngine.Snapshot import Snapshot


class GomokuRunner:

    def __init__(self, **kwargs) -> None:

        self.players = [None, None]
        self.engine = Gomoku(**kwargs)

    def _run(self):

        while not self.engine.isover():

            print(f"\n--- Turn {self.engine.turn}. Player {self.engine.player_idx} is playing ...")
            p = self.players[self.engine.player_idx]
            time_before_turn = perf_counter()

            player_action = p.play_turn(self)

            time_after_turn = perf_counter()
            dtime_turn = int((time_after_turn - time_before_turn) * 1000)
            print(f"Played in {dtime_turn} ms")

            self.engine.apply_action(player_action)
            self.engine.next_turn()
            print(f"Game board (np.ndarray shape: [0, ...] -> p1 / [1, ...] -> p2):\n{self.engine.board}\n")

    def run(self, players: list, init_snapshot: int = None, n_games: int = 1):

        self.players = players
        winners = []
        for i in range(n_games):
            print(f"\n\t[GomokuRunner: Start game n°{i+1}/{n_games}]\n")

            self.engine.init_game()
            if init_snapshot:
                Snapshot.update_from_snapshot(self.engine, init_snapshot)

            self._run()
            print(f"\n\t[GomokuRunner: Player {self.engine.winner} win game n°{i+1}/{n_games}]\n")

            for p in self.players:
                p.init()

            winner = players[self.engine.winner] if self.engine.winner >= 0 else self.engine.winner
            winners.append(f"P{self.engine.winner}: {str(winner)}")

        return winners
