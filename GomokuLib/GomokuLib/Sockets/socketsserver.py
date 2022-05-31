import time

import numpy as np

from UISocket import UISocket
import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65430  # Port to listen on (non-privileged ports are > 1023)

def listen(s):

    while True:

        data = s.get_recv_queue()
        for d in data:

            if d == "stop-order":
                print(f"RECV A STOP ORDER")
                raise Exception()

            elif d:
                print(f"Server: New d: {d}")


        msg = {'0123': np.random.randint(0, 2, (3, 3), dtype=np.int32)}
        s.add_sending_queue(msg)

        s.add_sending_queue("request")
        print(f"REQUEST")

        time.sleep(1)


if __name__ == "__main__":

    s = UISocket(as_server=True)
    s.start_sock_thread()

    try:
        listen(s)

    except:
        pass
    finally:

        # while len(s.send_queue):
        #     time.sleep(2)
        #     print(s.stats, len(s.send_queue), len(s.recv_queue))

        try:
            s.stop_sock_thread()
        except:
            pass

        print(s.stats, len(s.send_queue), len(s.recv_queue))
        s.disconnect()