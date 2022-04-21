from turtle import color
import pygame


class Button:

    def __init__(self, win: pygame.Surface,
                 origin: tuple, size: tuple,
                 event_code: str, color: tuple[int, int, int, [int]]):

        self.win = win
        self.origin = origin
        self.size = size
        self.ox, self.oy = self.origin
        self.dx, self.dy = self.size
        self.event_code = event_code
        self.color = color
        self.init_ui()

    def mouse_click(self, event):
        if (event.pos[0] < self.ox or event.pos[0] > self.ox + self.dx
            or event.pos[1] < self.oy or event.pos[1] > self.oy + self.dy):
            return None
        print(f"Click ! Code: {self.event_code}")
        return {
            'code': self.event_code,
        }

    def init_event(self, manager):
        manager.register(pygame.MOUSEBUTTONUP, self.mouse_click)

    def init_ui(self):
        button = pygame.Rect(self.origin, self.size)
        pygame.draw.rect(self.win, self.color, button)
        # self.win.blit(self.button, (self.ox + x - self.csx, self.oy + y - self.csy))

    def draw(self, **kwargs):
        pass
        # on_off_btn = pygame.Rect(self.panelx + marginx, self.panely + marginy, self.on_off_btn_width)