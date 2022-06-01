
"""

    Je vous avais dis que j'essayerai de finir le projet avant le 23 mai
    Mauvaises décisions au déb_ut
    Des semaines qu'on opti pour respecter les conditions du sujets
    avoir besoin de 2sem supp
    esperant qu'une fois fini je puisse ...


    pickle is terrible for sending numpy arrays, absolutely terrible.
    Picke is platform dependent, it's extremely inefficient for
    sending numpy arrays and numpy comes with really good
    serialization support in the module like tostring and fromstring
    but these are copy method

"""

import GomokuLib
import numpy as np
from GomokuLib.Game.UI import UIManagerSocket

from UISocket import UISocket

# def send():

#     # s = UISocket(host="192.168.1.6", as_client=True)
#     s = UISocket(as_client=True)
#     s.start_sock_thread()

#     try:
#         i = 0
#         while True:

#             data = s.get_recv_queue()
#             for d in data:

#                 if d == "request":
#                     s.add_sending_queue("response")

#                 if d:
#                     print(f"Client: New d: {d}\n")
#                     i += 1

#             if i == 5:
#                 s.add_sending_queue("stop-order")
#                 raise Exception()

#             time.sleep(0.1)

#     except:
#         pass
#     finally:
#         # while s.stats['send'] != s.stats['recv']:
#         #     time.sleep(2)
#         #     print(s.stats, len(s.send_queue), len(s.recv_queue))

#         try:
#             s.stop_sock_thread()
#         except:
#             pass

#         print(s.stats, len(s.send_queue), len(s.recv_queue))
#         s.disconnect()


def UI_program():

    # try:
    print(f"UI program start init")
    gui = UIManagerSocket(
        win_size=(1500, 1000)
    )
    print(f"UI program end init")
    gui()
        # uisock = UISocket(as_server=True, name="UIProgram")
        # uisock.start_sock_thread()
    #
    # except Exception as e:
    #     print(f"UI program exception: {e}")
        # uisock.stop_sock_thread()
        # uisock.disconnect()


if __name__ == "__main__":
    # send()
    UI_program()