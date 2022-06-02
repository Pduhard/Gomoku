import socket
import time
from .UISocket import UISocket


class UISocketClient(UISocket):

    """ Socket connection
            Set blocking = True avant la connection et apres
    """

    def __init__(self, name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name or "UISocketClient"

    def _init_socket(self):

        # if self.sock:
        #     self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(True)
        self._send = self.sock.sendall
        self._recv = self.sock.recv
        # print(f"UISocketClient: {self.name}: New socket")

    def connect(self):
        """Try to establish a connection to the server"""

        # print(f"self.connected: {self.connected}")
        if not self.connected:
            self._init_socket()
            print(f"UISocketClient: {self.name}: Attempt to connect at {(self.host, self.port)}")

            try:
                self.sock.connect((self.host, self.port))
                self.sock.settimeout(0.1)
                self.connected = True
                print(f"UISocketClient: {self.name}: Connected at {self.host}(port {self.port}) as client")

            except socket.error:
                time.sleep(0.1)
                return False
        
        return self.connected
