from .BasicRule import BasicRule

from .Capture import Capture

from .NoDoubleThrees import NoDoubleThrees

from .GameEndingCapture import GameEndingCapture, ForceWinPlayer, ForceWinOpponent

__all__ = [
    'BasicRule',
    'Capture',
    'GameEndingCapture',
    'NoDoubleThrees',
    'ForceWinPlayer',
    'ForceWinOpponent'
]

RULES=['opening', 'restricting', 'endturn', 'winning', 'nowinning']