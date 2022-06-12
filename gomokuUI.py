import GomokuLib
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start data fetching from GomokuGUIRunner for User Interface.')
    parser.add_argument('--host', action='store', nargs=1, default=None, type=str, help="Ip address of machine running GomokuGUIRunner")
    parser.add_argument('--port', action='store', nargs=1, default=None, type=int, help="An avaible port of machine running GomokuGUIRunner")
    parser.add_argument('--win_size', action='store', nargs=2, default=(1500, 1000), type=int, help="Set the size of the window")

    args = parser.parse_args()

    gui = GomokuLib.Game.UI.UIManager(
        win_size=args.win_size,
        host=args.host[0] if args.host else None,
        port=args.port[0] if args.port else None
    )

    print(f"\nUIProgram: Start a new UIManager")
    gui()


    # while True:
    #     # try:
    #         print(f"\nUIProgram: Start a new UIManager")
    #         gui()
    #     # except Exception as e:
    #     #     print(f"Exception raised:\n\t{e}")
