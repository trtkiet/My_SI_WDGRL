from util import *
import torch
from multiprocessing import Pool
import os
from model import WDGRL
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import cupy as cp
import time

if __name__ == '__main__':
    layers = 20
    parametric_iterations = 00
    a_cpu = np.ones((200, 200))
    b_cpu = np.ones((200, 200))
    a_gpu = cp.ones((200, 200))
    b_gpu = cp.ones((200, 200))
    start = time.time()
    c_cpu = None
    for i in range(parametric_iterations):
        for j in range(layers):
            c_cpu = np.dot(a_cpu, b_cpu)
            print(c_cpu)
            c_cpu = None
    print(f'CPU time: {time.time() - start}s')
    start = time.time()
    c_gpu = None
    for i in range(parametric_iterations):
        for j in range(layers):
            c_gpu = cp.dot(a_gpu, b_gpu) 
            c_gpu = None
    print(f'GPU time: {time.time() - start}s')