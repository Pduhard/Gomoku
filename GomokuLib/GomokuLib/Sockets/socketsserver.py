import time

from UISocket import UISocket
import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65430  # Port to listen on (non-privileged ports are > 1023)

def listen(s):

    while True:
        # data = s.listen()
        # data = s.recv()

        data = s.get_recv_queue()
        for d in data:

            if d:
                print(f"New d: {d}")
                s.add_sending_queue(d)
            else:
                print(f"EMPTY DATA ==========\n")

            if d == "stop-order":
                print(f"RECV A STOP ORDER")
                return

        # time.sleep(0.5)


if __name__ == "__main__":
    s = UISocket(as_server=True)
    s.start_sock_thread()

    try:
        listen(s)
    finally:

        print(s.stats, len(s.send_queue), len(s.recv_queue))
        while len(s.send_queue):
            time.sleep(2)
            print(s.stats, len(s.send_queue), len(s.recv_queue))

        try:
            s.stop_sock_thread()
        except:
            pass
        s.disconnect()