
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

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65430  # Port to listen on (non-privileged ports are > 1023)

def send():

    s = UISocket(as_client=True)

    try:
        for i in range(5):
            msg = {'0123': np.random.randint(0, 2, (3, 3), dtype=np.int32)}
            s.send(msg)
            # s.listen()
            time.sleep(1)

    finally:
        s.disconnect()


if __name__ == "__main__":
    send()