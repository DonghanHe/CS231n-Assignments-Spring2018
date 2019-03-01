import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  num_class = W.shape[1]
  num_inst, num_dim = X.shape
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_inst):

    XiW = X[i].dot(W)
    XiW -= np.max(XiW)
    exps = np.exp(XiW)
    sum_exps = np.sum(exps)
    loss += -np.log(exps[y[i]]/sum_exps)
    
    dW += 1 / sum_exps * X[i,np.newaxis].T*exps
    dW[:,y[i]] -= X[i,:]
  loss /= num_inst
  loss += reg* np.sum(W**2)
  dW /= num_inst
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  XW = X.dot(W)
  XW -= np.max(XW, axis = 1, keepdims = 1)
  exps = np.exp(XW)
  sum_exps = np.sum(exps,axis = 1, keepdims = 1)
  loss += -np.sum(np.log((exps/sum_exps)[np.arange(num_train),y]))
  loss /= num_train
  loss += reg * np.sum(W*W)
  
  ind = np.zeros_like(exps)
  ind[np.arange(num_train),y] = 1
  Q = (exps/sum_exps) - ind
  dW = X.T.dot(Q)
  dW /= num_train
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

