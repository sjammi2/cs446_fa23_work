import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def linear_kernel(x, y):
    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
    
    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        pass

def polynomial_kernel(x, y, p):
    '''
    Compute the polynomial kernel function with arbitrary power p

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        p: the power of the polynomial kernel
    
    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        pass

def gaussian_kernel(x, y, sigma):
    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        sigma: parameter sigma in rbf kernel
    
    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        pass

def svm_epoch_loss(alpha, x_train, y_train, kernel=linear_kernel):
    '''
    Compute the linear kernel function

    Arguments:
        alpha: 1d tensor with shape (N,), alpha is the trainable parameter in our svm 
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
    
    Returns:
        a torch.float32 scalar which is the loss function of current epoch
    '''
    pass

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=linear_kernel, c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is the linear kernel.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    pass

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=linear_kernel):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    idx = torch.nonzero(alpha,as_tuple=True)
    alpha_ = alpha[idx]
    x_ = x_train[idx]
    y_ = y_train[idx]
    if len(alpha_) == 0:
        return torch.zeros((x_test.shape[0],))
    id = alpha_.argmin()
    b = 1/y_[id]
    for i in range(len(alpha_)):
        b -= alpha_[i]*y_[i]*kernel(x_[i],x_[id])
    y_test = torch.zeros((x_test.shape[0],))
    for j in range(len(x_test)):
        y_test[j] = b
        for i in range(len(alpha_)):
            y_test[j] += alpha_[i]*y_[i]*kernel(x_[i],x_test[j])
    return y_test

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        - A 2D convolutional layer (torch.nn.Conv2d) with 7 output channels, with kernel size 3
        - A 2D maximimum pooling layer (torch.nn.MaxPool2d), with kernel size 2
        - A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, with kernel size 2
        - A fully connected (torch.nn.Linear) layer with 10 output features

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here

        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        pass

class DigitsConvNetv2(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        You can customize your network structure here as long as the input shape and output shape are as specified.

        '''
        super(DigitsConvNetv2, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here

        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        pass

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss

    return train_losses, test_losses
