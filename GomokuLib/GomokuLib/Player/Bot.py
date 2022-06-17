from GomokuLib.Algo.MCTSNjit import MCTSNjit

class Bot:

    def __init__(self, algorithm, time_requested: int = 0) -> None:
        self.algo = algorithm
        self.play_turn = self._play_njit_turn if isinstance(self.algo, MCTSNjit) else self._play_turn
        self.time_requested = time_requested

    def __str__(self):
        return f"Bot -> {self.algo.str()}"

    def init(self):
        self.algo.init()

    def _play_turn(self, runner) -> tuple[int]:
        return self.algo(runner.engine)[1]

    def _play_njit_turn(self, runner) -> tuple[int]:
        return self.algo.do_your_fck_work(runner.engine, 0, self.time_requested)
