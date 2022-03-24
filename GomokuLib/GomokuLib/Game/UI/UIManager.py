import pygame

from .Board import Board
from .Button import Button

# from GomokuLib.Game.GameEngine.GomokuGUI import GomokuGUI


class UIManager:

    # def __init__(self, gomokuGUI: GomokuGUI, win_size: tuple):
    def __init__(self, win_size: tuple, n_cells: int):

        # self.gomokuGUI = gomokuGUI
        self.win_size = win_size
        self.n_cells = n_cells
        self.events = []
        self.initUI()

    def initUI(self):

        print("UIManager: init GUI")
        pygame.init()
        self.win = pygame.display.set_mode(self.win_size)

        self.components = [
            Board(self.win, self.win_size, 0, 0, 0.66, 1, self.n_cells),
            # Board(self.win, win_size, 0.66, 0.5, 0.33, 0.5),
            # Button(self.win, win_size, 0.83, 0.25, 0.1, 0.1)
        ]

        for o in self.components:
            o.initUI()

    def __call__(self, get_turn_data): # Thread function

        print("UIManager __call__()\n")
        while True:
            turn_data = get_turn_data()
            self.drawUI(turn_data)
            self.handle_event()

    def drawUI(self, *args, **kwargs):
        for o in self.components:
            o.draw(*args, **kwargs)

    def handle_event(self):
        pass

    def get_events(self) -> list:
        return self.events
