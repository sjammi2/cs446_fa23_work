import torch 
import torch.matmul as matmul
import torch.tensor.shape as shape
import torch.tensor.t as t
import torch.cat as cat 
import torch.ones as ones
import torch.zeros as zeros
import torch.reshape as reshape

def linear_gd(X: tensor,Y: tensor, lrate: float, num_iter: int):
    ''' CS446 - Machine Learning - HW2
        Problem 4.1.a - Linear Regression
        implements linear regression using gradient descent
        Args:
            X: torch.tensor - shape (N, D)
            Y: torch.tensor - shape (N, 1)
            lrate: float - learning rate
            num_iter: int - number of iterations
        Returns:
            w: torch.tensor - shape (D + 1, 1)
    '''
    # initializing w as a zero vector of shape (D + 1, 1)
    w = zeros(shape(X)[1] + 1, 1)

    # adding a column of ones to X for the bias term in the beginning of the matrix to account for w0
    X = cat((ones(shape(X)[0], 1), X), 1) 

    # the gradient descent algorithm
    for i in range(num_iter):
        # calculating the gradient
        grad = 2 * matmul(X.T, matmul(X, w) - Y) / shape(X)[0]
        # updating w
        w = w - lrate * grad

    return w

def linear_normal(X: tensor,Y: tensor):
    ''' CS446 - Machine Learning - HW2
        Problem 4.1.b - Linear Regression
        implement linear regression using the pseudo inverse to solve for w
        Args:
            X: torch.tensor - shape (N, D)
            Y: torch.tensor - shape (N, 1)
        Returns:
            w: torch.tensor - shape (D + 1, 1)
    '''
    # adding a column of ones to X for the bias term in the beginning of the matrix to account for w0
    X = cat((ones(shape(X)[0], 1), X), 1)
    
    return matmul(matmul(X.T, X).inverse(), matmul(X.T, Y))

