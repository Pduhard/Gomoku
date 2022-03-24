import pygame


class Button:

    def __init__(self, win: pygame.Surface, win_size: tuple,
                 ox_prop: int, oy_prop: int,
                 wx_prop: int, wy_prop: int):

        self.win = win
        self.ox = win_size[0] * ox_prop
        self.oy = win_size[1] * oy_prop
        self.wx = win_size[0] * wx_prop
        self.wy = win_size[1] * wy_prop

    def draw(self, *args, **kwargs):
        pass
        # on_off_btn = pygame.Rect(self.panelx + marginx, self.panely + marginy, self.on_off_btn_width)