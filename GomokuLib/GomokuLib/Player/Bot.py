from GomokuLib.Algo.MCTSNjit import MCTSNjit
import faulthandler

class Bot:

    def __init__(self, algorithm) -> None:
        self.algo = algorithm
        self.play_turn = self._play_njit_turn if isinstance(self.algo, MCTSNjit) else self._play_turn

    def __str__(self):
        return f"Bot with algo: {str(self.algo)}"

    def _play_turn(self, runner) -> tuple[int]:
        return self.algo(runner.engine)[1]

    def _play_njit_turn(self, runner) -> tuple[int]:
        return self.algo.do_your_fck_work(runner.engine)
