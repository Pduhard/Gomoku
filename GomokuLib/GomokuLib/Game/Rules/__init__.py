from .AbstractRule import AbstractRule

from .BasicRule import BasicRule
from .BasicRuleJit import BasicRuleJit

from .Capture import Capture
from .CaptureJit import CaptureJit

from .NoDoubleThrees import NoDoubleThrees
from .NoDoubleThreesJit import NoDoubleThreesJit

from .GameEndingCapture import GameEndingCapture, ForceWinPlayer, ForceWinOpponent
from .GameEndingCaptureJit import GameEndingCaptureJit

__all__ = [
    'AbstractRule',
    'BasicRule',
    'BasicRuleJit',
    'Capture',
    'CaptureJit',
    'GameEndingCapture',
    'GameEndingCaptureJit',
    'NoDoubleThrees',
    'NoDoubleThreesJit',
    'ForceWinPlayer',
    'ForceWinOpponent'
]

RULES=['opening', 'restricting', 'endturn', 'winning', 'nowinning']