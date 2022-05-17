
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

import socket
import time

import numpy as np

from UISocket import UISocket

def send():

    # s = UISocket(host="192.168.1.6", as_client=True)
    s = UISocket(as_client=True)
    s.start_sock_thread()

    try:
        for i in range(20):
            msg = {'0123': np.random.randint(0, 2, (3, 3), dtype=np.int32)}

            s.send(msg)
            # s.add_sending_queue(msg)
            time.sleep(0.1)

        s.send("stop-order")
        # s.add_sending_queue("stop-order")
        time.sleep(2)

    finally:
        print(s.stats, len(s.send_queue), len(s.recv_queue))
        while s.stats['send'] != s.stats['recv']:
            time.sleep(2)
            print(s.stats, len(s.send_queue), len(s.recv_queue))

        try:
            s.stop_sock_thread()
        except:
            pass
        # print(s.get_recv_queue())

        s.disconnect()


def UI_program():

    try:
        print(f"UISocket GUI start init")
        uisock = UISocket(as_server=True, name="UIProgram")
        uisock.start_sock_thread()
        print(f"UISocket GUI end init")

    except Exception as e:
        print(f"UI program exception: {e}")
        uisock.stop_sock_thread()
        uisock.disconnect()


if __name__ == "__main__":
    # send()
    UI_program()