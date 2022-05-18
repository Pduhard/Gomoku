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
                 name: str = None):
        assert as_server or as_client

        self.host = host
        self.port = port
        self.as_server = as_server
        self.as_client = as_client
        self.name = name or ("Server" if self.as_server else "Client")

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

        self.sock_thread = False
        self.send_all_before_quit = False
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
        # print(f"UISocket: {self.name}: send() IN")
        try:
            self._send(self._serialize(data))
            self.stats['send'] += 1
            print(f"UISocket: {self.name}: Successfully sent.")

        except Exception as e:
            # print(f"UISocket: {self.name}: send() raise an Exception:\n{e}")
            self.connected = False
            raise Exception("send(): No connection yet ...")

        else:
            return True

    def recv(self):

        # print(f"UISocket: {self.name}: recv() IN")
        try:
            data = b''
            while True:
                buff = self._recv(self.BUFF_SIZE)
                # print(f"buff: {buff}")
                data += buff
                if len(buff) < self.BUFF_SIZE:
                    break

        except socket.error:
            # print(f"UISocket: {self.name}: socket.error: No data to recv")
            return None

        except Exception as e:
            # print(f"UISocket: {self.name}: recv() raise an Exception:\n{e}")
            self.connected = False
            raise Exception("recv(): No connection yet ...")

        else:
            if data:
                data = self._deserialize(data)
                self.stats['recv'] += 1
                print(f"UISocket: {self.name}: Data recv")
            return data

    def disconnect(self):
        # self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

    """ Server part """

    def _init_socket_as_server(self):

        # if self.sock:
        #     self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        # self.sock.setblocking(False)
        self.sock.settimeout(1)
        print(f"UISocket: {self.name}: New socket at {self.host} (port {self.port})")

    def _connect_as_server(self):
        """ Server looks for new connections to accept """

        try:
            self._init_socket_as_server()

            # print(F"Wait for accept connection to the server: {self.sock}.accept({(self.host, self.port)}")
            self.sock.listen()
            self.connection, self.addr = self.sock.accept()
            print(f"UISocket: {self.name}: New connection from client at {self.host} (addr {self.addr}, port {self.port})")

            self.connection.setblocking(False)

            self.connected = True
            self._send = self.connection.sendall
            self._recv = self.connection.recv

        # except socket.timeout:
        #     if not self.connection:
        #         print(f"UISocket: {self.name}: socket.timeout")
        #         self.connected = False
            # else:
            #     print(f"UISocket: {self.name}: No new connection")

        except socket.error:
            print(f"UISocket: {self.name}: socket.accept({(self.host, self.port)}): socket.error")
        #
        # except BlockingIOError:
        #     self.connected = False
        #     print(f"UISocket: {self.name}: BlockingIOError")

    """ Client part """

    def _init_socket_as_client(self):

        # if self.sock:
        #     self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.sock.settimeout(1)
        self.sock.setblocking(True)
        self._send = self.sock.sendall
        self._recv = self.sock.recv
        # print(f"UISocket: {self.name}: New socket")

    def _connect_as_client(self):
        """Try to establish a connection to the server"""

        while not self.connected:
            self._init_socket_as_client()
            print(f"UISocket: {self.name}: {time.time()}: Connection to {(self.host, self.port)}")

            try:
                self.sock.connect((self.host, self.port))
                self.connected = True

            except socket.error:
                time.sleep(0.5)
                pass

        print(f"UISocket: {self.name}: Connected at {self.host}(port {self.port}) as client")

    """ Thread part """

    def start_sock_thread(self):
        self.sock_thread = True
        # self.thread = threading.Thread(target=self._communication, daemon=True)
        self.thread = threading.Thread(target=self._communication)
        self.thread.start()

    def stop_sock_thread(self, send_all: bool = False):
        self.send_all_before_quit = send_all
        self.sock_thread = False
        print(f"UISocket: {self.name}: Stop Thread={self.sock_thread}: Send all data left ? -> {self.send_all_before_quit}")
        # while len(self.send_queue):
        #     print(f"UISocket: {self.name}: {len(self.send_queue)} more data to send")
        #     time.sleep(0.5)
        self.thread.join()
        print(f"UISocket: {self.name}: Thread.join() end")

    def _communication(self):
        while self.sock_thread or (self.send_all_before_quit and len(self.send_queue)):

            # if self.as_server:
            #     self.connect()  # Always search to connect with new client (No detection of client deco)
            if not self.connected:
                self.connect()

            try:
                if len(self.send_queue):
                    if self.send_queue_lock.acquire(timeout=0.1):
                        data = self.send_queue[0]
                        if self.send(data):
                            del self.send_queue[0]
                    self.send_queue_lock.release()
            except Exception as e:
                print(f"UISocket: {self.name}: _communication(): raise {e}")
                continue

            try:
                data = self.recv()
                if data:
                    if self.recv_queue_lock.acquire(timeout=0.1):
                        self.recv_queue.append(data)
                    self.recv_queue_lock.release()
            except Exception as e:
                print(f"UISocket: {self.name}: _communication(): raise {e}")
                continue

            # print(f"UISocket: {self.name}: will_send={len(self.send_queue)}, hav_recv={len(self.recv_queue)}")
            time.sleep(0.2)

        # while len(self.send_queue):
        #     print(f"UISocket: {self.name}: Last send ...")
        #     data = self.send_queue[0]
        #     if self.send(data):
        #         del self.send_queue[0]

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
