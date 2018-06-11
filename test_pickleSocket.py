
from threading import Thread
from PickleSocket import PickleSocket

import numpy as np
from tqdm import trange

def task():
    HOST = 'localhost'
    PORT = 50007
    INTEGER_LENGTH = 8
    with PickleSocket(HOST, PORT, INTEGER_LENGTH, as_server=False) as sock:
        for _ in range(10000):
            sock.send(np.random.rand(1024,1024))

if __name__ == '__main__':
    HOST = 'localhost'
    PORT = 50007
    INTEGER_LENGTH = 8
    
    process = Thread(target=task, daemon=True)
    process.start()
    
    with PickleSocket(host=HOST, port=PORT, integer_length=INTEGER_LENGTH) as sock:
        for _ in trange(10000):
            dict = sock.read()