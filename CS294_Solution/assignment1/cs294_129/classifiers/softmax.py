import numpy as np
from random import shuffle

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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #print(num_classes)
  #print(num_train)

  for i in range(num_train):
      linear_score = X[i,:].dot(W)
      #print(linear_score.shape)
      #print(linear_score.shape)
      # Normalization
      const = np.max(linear_score)
      score = linear_score - const # For numeric stability
      # Loss function
      sum_i = 0.0
      for t in score:
          #print(t)
          sum_i += np.exp(t)
      #print(sum_i.shape)
      loss += -score[y[i]] + np.log(sum_i)
      #print(loss.shape)


      # Gradient
      for j in range(num_classes):
          prob = np.exp(score[j])/sum_i
          #print("prob is: ", prob)
          #print("shape of prob: ", prob.shape)
          #print("shape of y: ", y[i].shape)
          #print("Shape of X[i, :]: ", X[i, :].shape)
          dW[:, j] += (prob - (j == y[i]))*X[i, :]

  loss /= num_train
  dW /= num_train

    # Adding regularization
  loss += 0.5*reg* (np.sum(W*W))
  dW += reg*W

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # My Code
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # Loss
  linear_score = X.dot(W)
  score = linear_score - np.max(linear_score)

  tmp = score[range(num_train), y]
  #print(tmp.shape)
  #print(np.log(np.exp(tmp)/np.sum(np.exp(score))).shape)
  #print(np.exp(score).shape)
  #print(np.sum(np.exp(score)).shape)
  loss = -np.mean(np.log(np.exp(tmp)/np.sum(np.exp(score), axis = 1)))

  # gradient
  prob = np.exp(score)/(np.sum(np.exp(score), axis=1).reshape(num_train, 1))
  #print("Shape of Interest: ", np.sum(np.exp(score), axis=1).shape)
  indx = np.zeros(prob.shape)
  indx[range(num_train), y] = 1
  #print(indx.shape)
  dW = np.dot(X.T, (prob - indx))
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  return loss, dW
