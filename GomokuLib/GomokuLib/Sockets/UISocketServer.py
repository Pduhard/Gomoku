import socket
import time
from .UISocket import UISocket

class UISocketServer(UISocket):

    """ Socket connection 
        self.sock.settimeout(0.1) on init
        self.connection.setblocking(False) after accept
        """

    def __init__(self, name: str = None, *args, **kwargs):
        print(f"\nUISocketServer: __init__(): START")

        super().__init__(*args, **kwargs)
        self.name = name or "UISocketServer"

        print(f"UISocketServer: __init__(): DONE")

    def _init_socket(self):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(1)

    def connect(self):
        """ Server looks for new connections to accept """

        if not self.connected:

            try:
                self._init_socket()

                self.sock.listen()
                self.connection, self.addr = self.sock.accept()
                print(f"UISocketServer: {self.name}: New connection from client at {self.host} (addr {self.addr}, port {self.port})")

                self.connection.setblocking(False)

                self.connected = True
                self._send = self.connection.sendall
                self._recv = self.connection.recv

            except socket.error:
                print(f"UISocketServer: {self.name}: Attempt to connect at {(self.host, self.port)}")
                return False
        
        return self.connected
