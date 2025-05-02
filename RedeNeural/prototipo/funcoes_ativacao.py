import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def sigmoid_derivada(s):
    ds = s * (1 - s)
    return ds

def tanh(z):
    t = np.tanh(z)
    return t

def tanh_derivada(t):
    dt = 1 - np.power(t, 2)
    return dt

def relu(z):
    a = np.maximum(0, z)
    return a

def relu_derivada(z):
    dz = np.int64(z > 0)
    return dz