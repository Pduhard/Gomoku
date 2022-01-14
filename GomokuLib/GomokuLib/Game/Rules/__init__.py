from .AbstractRule import AbstractRule
from .BasicRule import BasicRule
from .Capture import Capture
from .GameEndingCapture import GameEndingCapture
from .NoDoubleThrees import NoDoubleThrees

__all__ = [
    'AbstractRule',
    'BasicRule',
    'Capture',
	'GameEndingCapture',
	'NoDoubleThrees'
]

RULES=['opening', 'restricting', 'endturn', 'winning']