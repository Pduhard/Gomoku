from .Board import Board
from .Graph import Graph
from .Button import Button
from .Display import Display
from .UIManager import UIManager
from .HumanHints import HumanHints

import matplotlib
from matplotlib import pyplot as plt
print(plt.get_backend())
matplotlib.use('TkAgg')
print(plt.get_backend())

__all__ = [
    'Board',
    'Graph',
    'Button',
    'Display',
    'UIManager',
    'HumanHints'
]
