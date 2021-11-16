from tkinter import *

class GUIBoard(Frame):

    def __init__(self, size=19, win_size=(1000, 1000)):
        super().__init__()
        self.size = size
        self.win_size = win_size
        self.initUI()


    def initUI(self):
        wx, wy = self.win_size
        cxs, cys = wx / self.size, wy / self.size
        self.cell_size = (cxs, cys)
        # self.win_pad = (wpx, wpy)

        self.master.title("Go board %d x %d " % (self.size, self.size))
        self.pack(fill=BOTH, expand=1)
        canvas = Canvas(self)
        self.canvas = canvas

        for i in range(self.size):
            wpx, wpy = cxs / 2, cys / 2
            xi, yi = int(wpx + i * cxs), int(wpy + i * cys) 
            canvas.create_line(wpx, yi, wx - wpx, yi)
            canvas.create_line(xi, wpy,  xi, wy - wpy)
        canvas.bind('<Button-1>', self.click_callback)
        canvas.pack(fill=BOTH, expand=1)
    
    # @staticmethod
    def click_callback(self, event):
        cxs, cys = self.cell_size
        x0, y0 = event.x - (event.x % cxs), event.y - (event.y % cys)
        x1, y1 = x0 + cxs, y0 + cys
        self.canvas.create_oval(x0, y0, x1, y1)


def main():

    root = Tk()
    ex = GUIBoard()
    root.geometry("1000x1000+300+300")
    root.mainloop()


if __name__ == '__main__':
    main()