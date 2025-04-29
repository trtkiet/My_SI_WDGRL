from operations import *

TPB = 32
def Relu(a, b, x, itv):
    return GetReluInterval(a, b, x, itv)

def LinearWeight(a, b, x, weight):
    return MatMulMat(a, weight), MatMulMat(b, weight), MatMulMat(x, weight)

def LinearBias(a, b, x, bias):
    return MatAddBias(a, bias), b, MatAddBias(x, bias)