import numpy as np 
import torch 

def numpy_squares(k):
    return np.arange(k)**2

def torch_squares(k):
    return torch.arange(k)**2