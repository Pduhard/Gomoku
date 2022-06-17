from turtle import color
import pygame


class Button:

    def __init__(self, win: pygame.Surface,
                 origin: tuple,
                 size: tuple,
                 event_code: str,
                 color: tuple[int, int, int],
                 name: str = None,
                 num_states: int = 1):

        self.win = win
        self.origin = origin
        self.size = size
        self.event_code = event_code
        self.name = name or event_code
        self.state_max = num_states

        self.ox, self.oy = self.origin
        self.dx, self.dy = self.size
        self.text_size = int(self.dx / 9)
        self.color = color
        self.state = 0
        self.init_ui()

    def mouse_click(self, event):
        if (event.pos[0] < self.ox or event.pos[0] > self.ox + self.dx
            or event.pos[1] < self.oy or event.pos[1] > self.oy + self.dy):
            return None
        print(f"Button: Mouse click ! Code: {self.event_code}")

        self.state += 1
        if self.state == self.state_max:
            self.state = 0

        return {
            'code': self.event_code,
            'state': self.state
        }

    def init_event(self, manager):
        manager.register(pygame.MOUSEBUTTONUP, self.mouse_click)

    def init_ui(self):
        self.button = pygame.Rect(self.origin, self.size)

    def draw(self, **kwargs):
        pygame.draw.rect(self.win, self.color, self.button)
        self.blit_text(f"{self.name}: {self.state}")

    def blit_text(self, text):
        font = pygame.font.SysFont('arial', self.text_size)
        txt = font.render(text, True, (0, 0, 0))
        self.win.blit(txt, (self.ox + self.dx / 10, self.oy + self.dy / 2 - self.text_size / 2))
