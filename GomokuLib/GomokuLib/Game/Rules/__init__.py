from .AbstractRule import AbstractRule
# from .BasicRule import BasicRule, njit_is_align
from .BasicRule import BasicRule
from .BasicRuleJit import BasicRuleJit
from .Capture import Capture
from .NoDoubleThrees import NoDoubleThrees
from .NoDoubleThreesJit import NoDoubleThreesJit
from .GameEndingCapture import GameEndingCapture, ForceWinPlayer, ForceWinOpponent

__all__ = [
    'AbstractRule',
    'BasicRule',
    'BasicRuleJit',
    'Capture',
	'GameEndingCapture',
	'NoDoubleThrees',
    'NoDoubleThreesJit',
    'ForceWinPlayer',
    'ForceWinOpponent'
]

RULES=['opening', 'restricting', 'endturn', 'winning', 'nowinning']