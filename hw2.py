import torch
import hw2_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    w = torch.zeros(X.shape[1] + 1, 1)

    # adding a column of ones to X for the bias term in the beginning of the matrix to account for w0
    X = torch.cat((torch.ones(X.shape[0], 1), X), 1) 

    # the gradient descent algorithm
    for i in range(num_iter):
        # calculating the gradient
        #grad = 2 * torch.matmul(X.T, torch.matmul(X, w) - Y) / torch.shape(X)[0]
        # updating w
        g = ((2/torch.shape(X)[0]) * (torch.matmul(X, w) - Y).permute(1,0)) @ X
        
        w = w - lrate * g.permute(1,0)

    return w


def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    X = torch.cat((torch.ones(X.shape[0], 1), X), 1)
    #print(X.shape)
    #print(torch.matmul(X.T, X).shape)
    #print(torch.matmul(X.T, Y).shape)
    #print(torch.matmul(torch.matmul(X.T, X).inverse(), torch.matmul(X.T, Y)).shape)
    #return torch.matmul(torch.matmul(X.T, X).inverse(), torch.matmul(X.T, Y))
    return torch.matmul(torch.pinverse(X),Y) 

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    #generating the data
    X, Y = utils.load_reg_data()
    w = linear_normal(X, Y)

    # plotting the data
    plt.plot(X, torch.cat((torch.ones(X.shape[0], 1), X), 1) @ w, color = "red")
    plt.scatter(X, Y, color = "blue")
    plt.xlabel("data points")
    plt.ylabel("target values")
    plt.title("Linear Regression")
    return plt.gcf()


# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    w = torch.zeros(X.shape[1] + 1, 1)
    # prepending a column of ones to X to account for the bias term present in w0
    X = torch.cat((torch.ones(X.shape[0], 1), X), 1)

    # the gradient descent algorithm for logistic regression
    for i in range(num_iter):
        #calculating the gradient
        #grad = - ((X.T @ y) * (torch.exp(-y.T @ X @ w)) )/(X.shape[0] * (1 + (1/2)* torch.exp(-y.T @ X @ w)))
        g = - torch.exp(-Y * (torch.matmul(X,w)))/(2 + torch.exp(-Y * (torch.matmul(X,w)))) * Y * X
        w = w - lrate * (1/X.shape[0])*torch.sum(g, dim = 0).unsqueeze(1)
    return w


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    # generating the data
    X, Y = utils.load_logistic_data()
    w_n = linear_normal(X, Y)
    w_l = logistic(X, Y)
    # plotting the data
    plt.plot(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], 'x', label="Y = 1")
    plt.plot(X[Y[:,0] == -1, 0], X[Y[:,0] == -1, 1], 'x', label="Y = -1")
    # plotting the decision boundary for logistic regression
    a = torch.linspace(X[:,0].min(), X[:,0].max(), 100)
    b = - (w_l[0] + w_l[1] * a) / w_l[2]
    plt.plot(a, b, label="logistic regression boundary line")
    # plotting the decision boundary for linear regression
    c = - (w_n[0] + w_n[1] * a) / w_n[2]
    plt.plot(a, c, label="logistic regression boundary line")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(bbox_to_anchor =(0.75, 1.15))
    plt.title("Logistic Regression boundary line vs Linear Regression boundary line")
    plt.gcf()
