import socket
import pickle
import time
import numpy as np

class UISocket:

    BUFF_SIZE: int = 4096

    def __init__(self, host: str = None, port: int = None):

        self.host = host or "localhost"
        self.port = port or 31415

        self.sock = None
        self.addr = None
        self.connection = None
        self.connected = False

        self.send_all_before_quit = False
        self.send_queue = []

        self.stats = {
            'send': 0,
            'recv': 0,
        }

    def connect(self):
        raise NotImplementedError

    def add_sending_queue(self, data):
        print(f"UISocket: {self.name}: Add sending queue")
        self.send_queue.append(data)

    def _serialize(self, data):
        b = pickle.dumps(data, -1)
        return b

    def _deserialize(self, data):
        obj = pickle.loads(data)
        return obj


    """ Sending part """

    def _send(self, *args):
        raise NotImplementedError

    def send(self):
        try:
            if self.connect():

                data = self.send_queue[0]
                self._send(self._serialize(data))

                self.stats['send'] += 1
                del self.send_queue[0]
                print(f"UISocket: {self.name}: Successfully sent.")

        except Exception as e:
            self.connected = False
            print(f"UISocket: {self.name}: send() failed: {type(e)}: {e}")
            return False

        return self.connected

    def send_all(self, force=False):

        while len(self.send_queue):
            if not self.send():
                print(f"Try to send {len(self.send_queue)} data left")
                if not force:
                    break

            time.sleep(0.1)

    """ Receive part """

    def _recv(self, *args):
        raise NotImplementedError

    def recv(self):

        # print(f"UISocket: {self.name}: recv() IN")
        try:
            if self.connect():

                data = b''
                while True:
                    buff = self._recv(self.BUFF_SIZE)
                    # print(f"buff: {buff}")
                    data += buff
                    if len(buff) < self.BUFF_SIZE:
                        break

                if data:
                    data = self._deserialize(data)
                    self.stats['recv'] += 1
                    print(f"UISocket: {self.name}: Data recv")
                return data
    
        except socket.error:
            pass
            # print(f"UISocket: {self.name}: socket.error: No data to recv")

        except Exception as e:
            print(f"UISocket: {self.name}: recv() failed: {type(e)}: {e}")
            self.connected = False
        
        return None


    # def recv_all(self):
    #     """ Useless ?"""

    #     queue = []
    #     try:
    #         data = True
    #         while data:
    #             data = self.recv()
    #             if data:
    #                 queue.append(data)

    #     except Exception as e:
    #         print(f"UISocket: {self.name}: {e}")

    #     return queue

    def disconnect(self):
        # self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()






# from GomokuLib.Sockets.UISocketClient import UISocketClient
# from GomokuLib.Sockets.UISocketServer import UISocketServer

# if __name__ == "__init__":

#     sockclient = UISocketClient()
#     sockserver = UISocketServer()

#     sockclient.c
