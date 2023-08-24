import numpy as np 
import torch 

def numpy_squares(k):
    return np.square(np.arange(k))

def torch_squares(k):
    return torch.square(torch.arange(k))