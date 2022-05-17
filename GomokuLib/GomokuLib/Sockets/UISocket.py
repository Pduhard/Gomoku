import errno
import socket
import pickle
import select
import threading
import time

import numpy as np


# class LockQueue(list):



class UISocket:

    BUFF_SIZE: int = 4096

    def __init__(self, host="localhost", port=31436, as_server=False, as_client=False):
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
            print("Thread: Successfully sent.")

        except Exception as e:
            print(f"UISocket.send() raise an Exception:\n{e}")
            self.connected = False
            self.connect()
            return False

        return True

    def recv(self):
        if self.as_server:
            self.connect()      # Need to connect with new client/server

        try:

            data = b''
            while True:
                buff = self._recv(self.BUFF_SIZE)
                # print(f"buff: {buff}")
                data += buff
                if len(buff) < self.BUFF_SIZE:
                    break

        # except socket.error as e:
        #     err = e.args[0]
        #     if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
        #         print(f"UISocket.recv() raise EAGAIN or EWOULDBLOCK\n{e}")
        #     self.connected = False
        #     self.connect()
        #     return None

        except Exception as e:
            # print(f"UISocket.recv() raise an Exception:\n{e}")
            self.connected = False
            self.connect()
            return None

        if data:
            data = self._deserialize(data)
            self.stats['recv'] += 1
            print(f"Thread: Data recv")
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
        self.sock.settimeout(0.5)
        self.sock.setblocking(False)
        print(f"New socket at {self.host} (port {self.port})")

    def _connect_as_server(self):
        """ Server looks for new connections to accept """

        try:
            # print("Wait for accept connection to the server")
            self.connection, self.addr = self.sock.accept()
            print(f"New connection from client at {self.host} (addr {self.addr}, port {self.port})")

            self.connection.setblocking(False)

            self.connected = True
            self._send = self.connection.sendall
            self._recv = self.connection.recv

        except socket.timeout:
            if not self.connection:
                print(f"No connection accepted yet ...")
                self.connected = False
            # else:
            #     print(f"No new connection")

        except BlockingIOError:
            self.connected = False
            # print(f"No new connection")

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
        print("Try to reconnect ...", end='')
        while not self.connected:

            try:
                self.sock.connect((self.host, self.port))
                self.connected = True

            except socket.error:
                self.connected = False
                print(".")
                time.sleep(1)

        print(f"Connected at {self.host}(port {self.port}) as client")

    """ Thread part """

    def start_sock_thread(self):
        self.sock_thread = True
        self.thread.start()

    def stop_sock_thread(self):
        self.sock_thread = False
        # print("Thread.join() begin")
        print("Thread.join() end")
        while len(self.send_queue):
            print(f"Thread: had {len(self.send_queue)} more data to send")
        self.thread.join()

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
                print(f"GROS PROBLEME DANS LE SEND E DU THREAD")

            try:
                data = self.recv()
                if data:
                    self.recv_queue_lock.acquire()
                    self.recv_queue.append(data)
                    self.recv_queue_lock.release()
            except:
                print(f"GROS PROBLEME DANS LE RECV E DU THREAD")

            # print(f"Thread: will_send={len(self.send_queue)}, hav_recv={len(self.recv_queue)}")

        while len(self.send_queue):
            print("Thread: Last send ...")
            data = self.send_queue[0]
            if self.send(data):
                del self.send_queue[0]

        print("Thread: END COMMUNICATIONS")

    def get_recv_queue(self):
        self.recv_queue_lock.acquire()
        q = self.recv_queue
        self.recv_queue = []
        self.recv_queue_lock.release()
        return q

    def add_sending_queue(self, data):
        print("Add sending queue")
        self.send_queue_lock.acquire()
        self.send_queue.append(data)
        self.send_queue_lock.release()
