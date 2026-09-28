import torch.nn as nn

def ListWrapper(sep, type):
    
    def wrapper(string: str):
        return str2list(string, sep, type)
    
    return wrapper

def str2list(string: str, sep=',', type=int):
    a = [type(s.strip()) for s in string.split(sep)]
    return a if len(a) > 1 else a[0]

def str2activation(input: str):
    if input == 'relu':
        return nn.ReLU
    if input == 'elu':
        return nn.ELU
    if input == 'lrelu':
        return nn.LeakyReLU
    if input == 'tanh':
        return nn.Tanh
    if input == 'sigmoid':
        return nn.Sigmoid
    raise ValueError(f"Input {input} is not one of the valid activation functions: relu, elu, lrelu, tanh, or sigmoid")