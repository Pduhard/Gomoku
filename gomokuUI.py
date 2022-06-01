import GomokuLib
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start data fetching from GomokuGUIRunner for User Interface.')
    parser.add_argument('--host', action='store', nargs=1, default=None, type=str, help="Ip address of machine running GomokuGUIRunner")
    parser.add_argument('--port', action='store', nargs=1, default=None, type=int, help="An avaible port of machine running GomokuGUIRunner")

    args = parser.parse_args()

    gui = GomokuLib.Game.UI.UIManagerSocket(
        win_size=(1500, 1000),
        host=args.host[0] if args.host else None,
        port=args.port[0] if args.port else None
    )

    while True:
        try:
            print(f"UIProgram: Start a new UIManager")
            gui()
        except Exception as e:
            print(f"Exception raised:\n\t{e}")
