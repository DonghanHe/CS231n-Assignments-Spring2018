import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    up_count = 0
    for j in xrange(num_classes):
      
      if j == y[i]:
        continue
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        up_count +=1
        loss += margin
        dW[:,j] += X[i]

    dW[:,y[i]] -= up_count*X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  # Same for gradient
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  scores = X.dot(W)
  correct_scores = scores[np.arange(num_train),y] # note slicing this way
                                                  # returns a list, not a mat
  margins = np.maximum(0, scores - correct_scores[:,np.newaxis] + 1)
  margins[np.arange(num_train),y] = 0
  X_mask = (margins>0).astype(int)
  X_mask[np.arange(num_train),y] = -np.sum(X_mask, axis = 1)
  dW += X.T.dot(X_mask)
  loss = np.sum(margins)
  loss /= num_train
  dW /=num_train
  loss += reg * np.sum(W**2)
  dW += 2 * reg * W
  

  return loss, dW
