import numpy as np
import random
import math

from time import time
from ReplayMemory.SumTree import SumTree
    
if __name__ == '__main__':
    array = np.random.rand(5000, 80**2)
    begin = time()
    for _ in range(1000):
        b = random.sample(list(array), 32)
    print('Time Elapsed: {!s}'.format(time()-begin))
    array = [np.random.rand(80**2) for _ in range(20000)]
    begin = time()
    for _ in range(1000):
        b = random.sample(array, 32)
    print('Time Elapsed: {!s}'.format(time()-begin))
    s = sum(range(20000))
    weights = [i/s for i in range(20000)]
    begin = time()
    for _ in range(1000):
        b = random.choices(array, weights=weights, k=32)
    print('Time Elapsed: {!s}'.format(time()-begin))
    tree = SumTree(20000)
    for w in weights:
        tree.insert(np.random.rand(80**2), w)
    begin = time()
    for _ in range(1000):
        b = [tree.choose() for _ in range(32)]
    print('Time Elapsed: {!s}'.format(time()-begin))