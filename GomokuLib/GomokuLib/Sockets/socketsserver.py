from UISocket import UISocket
import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65430  # Port to listen on (non-privileged ports are > 1023)
BUFF_SIZE = 4096

def listen():

    s = UISocket()
    s.start_server()

    while True:
        data = s.listen()
        print(f"New data: {data}")

if __name__ == "__main__":
    listen()