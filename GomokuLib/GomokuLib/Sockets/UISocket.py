import socket
import pickle
import select
import time


class UISocket:

    BUFF_SIZE: int = 4096

    def __init__(self, host="localhost", port=31415, start_server=False, connect_to=False):
        self.host = host
        self.port = port
        self._start_server = start_server
        self._connect_to = connect_to

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = None
        self.connection = None
        self.connected = False

        if self._start_server:
            self.start_server()
        elif self._connect_to:
            self.connect()

    def start_server(self):
        """ Start the server and wait for the first connection"""
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.sock.settimeout(0.1)
        while not self.connection:
            self.fetch_connection()
        print(f"Connected at {self.host}(addr {self.addr}, port {self.port}) as server")

    def fetch_connection(self):
        """ Server looks for new connections to accept """
        try:
            connection, addr = self.sock.accept()
        except socket.timeout:
            print(f"No connection accepted yet ... (Connection exist: {bool(self.connection)})")
        else:
            self.connection, self.addr = connection, addr

    def connect(self):
        """Try to establish a connection to the server"""

        self.connected = False
        while not self.connected:
            print("Try to reconnect ...")
            # Attempt to reconnect, otherwise sleep
            try:
                self.sock.connect((self.host, self.port))
                self.connected = True
            except socket.error:
                time.sleep(1)

        print(f"Connected at {self.host}(port {self.port}) as client")

    def disconnect(self):
        self.sock.close()

    def _serialize(self, data):
        b = pickle.dumps(data, -1)
        # print(f"Serialize {data}\nTo {b}\n")
        return b

    def _deserialize(self, data):
        obj = pickle.loads(data)
        return obj

    def send(self, data):

        try:
            self.sock.sendall(self._serialize(data))
            print("Successfully sent.")
        except socket.error:
            print("socket.error")
            exit(0)
            self.connect()

    def recv(self):
        assert self.connection

        data = b''
        while True:
            buff = self.connection.recv(self.BUFF_SIZE)
            data += buff
            if len(buff) < self.BUFF_SIZE:
                break

        if data:
            data = self._deserialize(data)
        # print(f"Recv data {data}")
        return data

    def listen(self):
        data = None
        while not data:
            self.fetch_connection()  # Always try to accept new connection
            data = self.recv()
        return data
