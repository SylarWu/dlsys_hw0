import gzip
import struct

import numpy as np

try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as image_file:
        data_magic, data_nums, data_rows, data_cols = struct.unpack('>IIII', image_file.read(16))
        print(f"image data meta info: Magic:{data_magic}, Nums:{data_nums}, Rows:{data_rows}, Cols:{data_cols}")
        data_buffer = image_file.read(data_nums * data_rows * data_cols)
        data = np.frombuffer(data_buffer, dtype=np.uint8)
        data = np.reshape(data, (data_nums, data_rows * data_cols)).astype(np.float32)
        data = data / 255.0
    with (gzip.open(label_filename, 'rb')) as label_file:
        label_magic, label_nums = struct.unpack('>II', label_file.read(8))
        print(f"image label meta info: Magic:{label_magic}, Nums:{label_nums}")
        label_buffer = label_file.read(label_nums)
        label = np.frombuffer(label_buffer, dtype=np.uint8)
    return data, label
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    batch_size = Z.shape[0]
    log_of_sum_exp = np.log(np.sum(np.exp(Z), axis=-1))
    cor_prediction = Z[np.arange(batch_size), y]
    return np.mean(log_of_sum_exp - cor_prediction)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    assert X.shape[0] == y.shape[0]
    num_examples = X.shape[0]
    for i in range(0, num_examples, batch):
        x_batch = X[i: i + batch]
        y_batch = y[i: i + batch]

        exp_of_pred = np.exp(np.dot(x_batch, theta))
        z_batch = exp_of_pred / np.sum(exp_of_pred, axis=-1, keepdims=True)
        Iy = np.zeros_like(z_batch)
        Iy[np.arange(batch), y_batch] = 1.0

        grad = np.dot(x_batch.T, z_batch - Iy)

        theta -= ((lr / batch) * grad)
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """

    ### BEGIN YOUR CODE
    def normalize(matrix):
        exp = np.exp(matrix)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def to_one_hot(matrix, num_classes):
        ret = np.zeros((matrix.shape[0], num_classes))
        ret[np.arange(matrix.shape[0]), matrix] = 1.0
        return ret

    assert X.shape[0] == y.shape[0]
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    for i in range(0, num_examples, batch):
        # batch_size, input_dim
        x_batch = X[i: i + batch]
        # batch_size,
        y_batch = y[i: i + batch]

        # batch_size, hidden_dim
        z_batch = np.maximum(np.dot(x_batch, W1), 0)
        # batch_size, num_classes
        g_2 = normalize(np.dot(z_batch, W2)) - to_one_hot(y_batch, num_classes)
        # batch_size, hidden_dim
        g_1 = np.where(z_batch > 0, 1, 0) * np.dot(g_2, W2.T)

        W1 -= ((lr / batch) * np.dot(x_batch.T, g_1))
        W2 -= ((lr / batch) * np.dot(z_batch.T, g_2))
        ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1, cpp=True)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
