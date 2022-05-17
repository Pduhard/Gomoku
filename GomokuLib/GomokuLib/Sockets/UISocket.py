import errno
import socket
import pickle
import select
import threading
import time

import numpy as np


# class LockQueue(list):



class UISocket:
    """
        Passer la connection au debut de la loop _communication
        Plus de connect dans les except, seulement des self.connected = False
        Tenter de pas recréer la ptn de socket
    """

    BUFF_SIZE: int = 4096

    def __init__(self, host: str = "localhost", port: int = 31415,
                 as_server: bool = False, as_client: bool = False,
                 name: str = "Thread"):
        assert as_server or as_client

        self.host = host
        self.port = port
        self.as_server = as_server
        self.as_client = as_client
        self.name = name

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
            self.connect = self._connect_as_client
        self.connect()

        # self.thread = threading.Thread(target=self._communication, daemon=True)
        self.thread = threading.Thread(target=self._communication)
        self.sock_thread = False

        self.send_queue = []
        self.recv_queue = []
        self.send_queue_lock = threading.Lock()
        self.recv_queue_lock = threading.Lock()

        self.stats = {
            'send': 0,
            'recv': 0,
        }

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
            self.stats['send'] += 1
            print(f"UISocket: {self.name}: Successfully sent.")

        except Exception as e:
            print(f"UISocket: {self.name}: send() raise an Exception:\n{e}")
            self.connected = False
            self.connect()
            return False

        return True

    def recv(self):
        if self.as_server or not self.connected:
            self.connect()      # Need to connect with new client/server

        try:

            data = b''
            while True:
                buff = self._recv(self.BUFF_SIZE)
                # print(f"buff: {buff}")
                data += buff
                if len(buff) < self.BUFF_SIZE:
                    break

        except Exception as e:
            print(f"UISocket: {self.name}: recv() raise an Exception:\n{e}")
            self.connected = False
            # self.connect()
            return None

        if data:
            data = self._deserialize(data)
            self.stats['recv'] += 1
            print(f"UISocket: {self.name}: Data recv")
        return data

    def disconnect(self):
        self.sock.close()
        # self.sock.shutdown(socket.SHUT_RDWR)

    """ Server part """

    def _init_socket_as_server(self):

        if self.sock:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.sock.settimeout(0.1)
        self.sock.setblocking(False)
        print(f"UISocket: {self.name}: New socket at {self.host} (port {self.port})")

    def _connect_as_server(self):
        """ Server looks for new connections to accept """

        try:
            # print("Wait for accept connection to the server")
            self.connection, self.addr = self.sock.accept()
            print(f"UISocket: {self.name}: New connection from client at {self.host} (addr {self.addr}, port {self.port})")

            self.connection.setblocking(False)

            self.connected = True
            self._send = self.connection.sendall
            self._recv = self.connection.recv

        except socket.timeout:
            if not self.connection:
                print(f"UISocket: {self.name}: No connection accepted yet ...")
                self.connected = False
            # else:
            #     print(f"UISocket: {self.name}: No new connection")

        except BlockingIOError:
            self.connected = False
            # print(f"UISocket: {self.name}: No new connection")

    """ Client part """

    def _init_socket_as_client(self):

        if self.sock:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self._send = self.sock.sendall
        self._recv = self.sock.recv
        # print(f"UISocket: {self.name}: New socket")

    def _connect_as_client(self):
        """Try to establish a connection to the server"""

        self._init_socket_as_client()
        print(f"UISocket: {self.name}: Try to reconnect ...", end='')
        while not self.connected:

            try:
                self.sock.connect((self.host, self.port))
                self.connected = True

            except socket.error:
                self.connected = False
                print(".")
                time.sleep(1)

        print(f"UISocket: {self.name}: Connected at {self.host}(port {self.port}) as client")

    """ Thread part """

    def start_sock_thread(self):
        self.sock_thread = True
        self.thread.start()

    def stop_sock_thread(self):
        self.sock_thread = False
        # while len(self.send_queue):
        #     print(f"UISocket: {self.name}: {len(self.send_queue)} more data to send")
        #     time.sleep(0.5)
        self.thread.join()
        print(f"UISocket: {self.name}: Thread.join() end")

    def _communication(self):
        while self.sock_thread:

            try:
                if len(self.send_queue):
                    if self.send_queue_lock.acquire(timeout=0.1):
                        data = self.send_queue[0]
                        if self.send(data):
                            del self.send_queue[0]
                        self.send_queue_lock.release()
            except:
                print(f"UISocket: {self.name}: GROS PROBLEME DANS LE SEND E DU THREAD")

            try:
                data = self.recv()
                if data:
                    self.recv_queue_lock.acquire()
                    self.recv_queue.append(data)
                    self.recv_queue_lock.release()
            except:
                print(f"UISocket: {self.name}: GROS PROBLEME DANS LE RECV E DU THREAD")

            print(f"UISocket: {self.name}: will_send={len(self.send_queue)}, hav_recv={len(self.recv_queue)}")
            time.sleep(0.5)

        while len(self.send_queue):
            print(f"UISocket: {self.name}: Last send ...")
            data = self.send_queue[0]
            if self.send(data):
                del self.send_queue[0]

        print(f"UISocket: {self.name}: END COMMUNICATIONS")

    def get_recv_queue(self):
        self.recv_queue_lock.acquire()
        q = self.recv_queue
        self.recv_queue = []
        self.recv_queue_lock.release()
        return q

    def add_sending_queue(self, data):
        print(f"UISocket: {self.name}: Add sending queue")
        self.send_queue_lock.acquire()
        self.send_queue.append(data)
        self.send_queue_lock.release()
