import socket
import pickle
import select
import time


class UISocket:

    BUFF_SIZE: int = 4096

    def __init__(self, host="localhost", port=31415, as_server=False, as_client=False):
        assert as_server or as_client

        self.host = host
        self.port = port
        self.as_server = as_server
        self.as_client = as_client

        self.sock = None
        self.addr = None
        self.connection = None
        self.connected = False
        self._send = None
        self._recv = None

        if self.as_server:
            self._init_socket_as_server()
            self.connect = self._connect_as_server
        else:
            self._init_socket_as_client()
            self.connect = self._connect_as_client
        self.connect()

    """ Common part """

    def _serialize(self, data):
        b = pickle.dumps(data, -1)
        # print(f"Serialize {data}\nTo {b}\n")
        return b

    def _deserialize(self, data):
        obj = pickle.loads(data)
        return obj

    def send(self, data):
        try:
            self._send(self._serialize(data))
            print("Successfully sent.")

        except Exception as e:
            print(f"UISocket.send() raise an Exception:\n{e}")
            self.connected = False
            self.connect()

    def recv(self):
        try:
            data = b''
            while True:
                buff = self._recv(self.BUFF_SIZE)
                # print(f"buff: {buff}")
                data += buff
                if len(buff) < self.BUFF_SIZE:
                    break

        except Exception as e:
            print(f"UISocket.recv() raise an Exception:\n{e}")
            self.connected = False
            self.connect()
            return None

        if data:
            data = self._deserialize(data)
            print(f"Data recv")
        return data

    def listen(self):
        data = None
        while not data:
            self.connect()      # Need to connect with new client/server
            data = self.recv()
        return data

    def disconnect(self):
        self.sock.close()

    """ Server part """

    def _init_socket_as_server(self):

        if self.sock:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.sock.settimeout(0.5)
        # self.sock.setblocking(False)
        print(f"New socket at {self.host} (port {self.port})")

    def _connect_as_server(self):
        """ Server looks for new connections to accept """

        try:
            print("Wait for accept connection to the server")
            self.connection, self.addr = self.sock.accept()
            print(f"New connection from client at {self.host} (addr {self.addr}, port {self.port})")

            self.connected = True
            self._send = self.connection.sendall
            self._recv = self.connection.recv

        except socket.timeout:
            if not self.connection:
                print(f"No connection accepted yet ...")
                self.connected = False
            else:
                print(f"No new connection")

        except BlockingIOError:
            self.connected = False
            print(f"No new connection")

    """ Client part """

    def _init_socket_as_client(self):

        if self.sock:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._send = self.sock.sendall
        self._recv = self.sock.recv
        print(f"New socket")

    def _connect_as_client(self):
        """Try to establish a connection to the server"""

        self._init_socket_as_client()

        i = 0
        print("Try to reconnect ...", end='')
        while not self.connected:

            try:
                self.sock.connect((self.host, self.port))
                self.connected = True

            except socket.error:
                self.connected = False
                if i % 10:
                    print(".", end='')
                else:
                    print(".")
                time.sleep(0.1)
                i += 1

        print(f"Connected at {self.host}(port {self.port}) as client")
