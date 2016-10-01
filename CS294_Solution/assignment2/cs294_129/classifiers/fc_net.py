import numpy as np

from cs294_129.layers import *
from cs294_129.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a layer dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3*32*32, layer_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - layer_dim: An integer giving the size of the layer layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.randn(input_dim, layer_dim) * weight_scale
    self.params['b1'] = np.zeros(layer_dim)
    self.params['W2'] = np.random.randn(layer_dim , num_classes) * weight_scale
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    #####----------------------- Initialization ---------------------------#####
    W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
    # Feed Forward into the first and second layers
    layer_layer, cache_layer_layer = affine_relu_forward(X, W1, b1)
    scores, cache_scores = affine_forward(layer_layer, W2, b2)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #######--------------------- My Code Begins ---------------------######
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5*self.reg*np.sum(W1**2) + 0.5*self.reg*np.sum(W2**2)
    loss = data_loss + reg_loss

    #################################################################
    #######----------------- Backpropagation -----------------#######
    #################################################################
    # Second layer
    dx1, dW2, db2 = affine_backward(dscores, cache_scores)
    dW2 += self.reg*W2
    # First Layer
    dx, dW1, db1 = affine_relu_backward(dx1, cache_layer_layer)
    dW1 += self.reg*W1

    grads.update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of layer layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, layer_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - layer_dims: A list of integers giving the size of each layer layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(layer_dims)
    self.dtype = dtype
    self.params = {}
    self.cache = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    ####################################################################
    ###### ------------------------ My Code ----------------------######
    ####################################################################
    dims = [input_dim] + layer_dims + [num_classes]
    self.L = len(layer_dims) + 1

    #print(dims)
    #for i in xrange(self.num_layers):
    #    if i == self.num_layers - 1:
    #        self.params['b%d' % (i+1)] = np.zeros(dims[i+1])
    #        #print(self.params['b%d' % (i+1)])
    #        self.params['W%d' % (i+1)] = np.random.randn(dims[i], dims[i+1])*weight_scale
    #        #print(self.params['W%d' % (i+1)])
    #        # Note that we do not usually use batchnormalization in the final layer
    #    elif i == 0:
    #        #print("I am in this loop! Take a look here!")
    #        self.params['b%d' % (i+1)] = np.zeros(layer_dims[0])
    #        #print(self.params['b%d' % (i+1)])
    #        self.params['W%d' % (i+1)] = np.random.randn(dims[i], dims[i+1])*weight_scale
    #        #print(self.params['W%d' % (i+1)])
    #        if self.use_batchnorm:
    #            self.params['gamma' + repr(i+1)] = np.ones(layer_dims[i])
    #            self.params['beta' + repr(i+1)] = np.zeros(layer_dims[i])
    #    else:
    #        self.params['b%d' % (i+1)] = np.zeros(dims[i+1])
    #        #print(self.params['b%d' % (i+1)])
    #        self.params['W%d' % (i+1)] = np.random.randn(dims[i], dims[i+1])*weight_scale
    #        #print(self.params['W%d' % (i+1)])
    #        if self.use_batchnorm:
    #            self.params['gamma' + repr(i+1)] = np.ones(layer_dims[i])
    #            self.params['beta' + repr(i+1)] = np.zeros(layer_dims[i])

    #print(self.params['W1'])
    W = {'W' + repr(i+1): weight_scale * np.random.randn(dims[i], dims[i+1]) for i in range(len(dims)-1)}
    b = {'b' + repr(i+1): np.zeros(dims[i+1]) for i in range(len(dims) - 1)}

    self.params.update(b)
    self.params.update(W)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = {'bn_param' + repr(i+1): {'mode': 'train',
      'running_mean': np.zeros(dims[i+1]),
      'running_var': np.zeros(dims[i+1])} for i in xrange(len(dims) - 2)}
      gammas = {'gamma' + repr(i + 1): np.ones(dims[i + 1]) for i in range(len(dims)-2)}
      betas = {'beta' + str(i+1): np.zeros(dims[i+1]) for i in range(len(dims) - 2)}

      self.params.update(betas)
      self.params.update(gammas)
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    #layer = {}
    #layer[0] = X
    #cache_layer = {}

    #for i in range(1, self.num_layers):
    #    #print("Forward Pass!")
    #    layer[i], cache_layer['affine'+repr(i)] = affine_forward(layer[i-1],
    #                                                self.params['W%d' % i],self.params['b%d' % i])
    #    if self.use_batchnorm:
    #        #print('gamma is: ', self.params['gamma' + repr(i)])
    #        #print('beta is: ', self.params['beta' + repr(i)])
    #        # (self.bn_params[i])
    #        layer[i], cache_layer['bnorm'+repr(i)] = batchnorm_forward(layer[i], self.params['gamma' + repr(i)],
    #                                                                    self.params['beta' + repr(i)], {'mode': 'train'})
    #    layer[i], cache_layer['relu' + repr(i)] = relu_forward(layer[i])
    #    if self.use_dropout:
    #        layer[i], cache_layer['dropout' + repr(i)] = dropout_forward(layer[i],
    #                    self.dropout_param)

    # feed forward


    # We are gonna store everythin in a dictionnary layer
    layer = {}
    layer['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    if self.use_dropout:
        # dropout on the input layer
        hdrop, cache_hdrop = dropout_forward(
            layer['h0'], self.dropout_param)
        layer['hdrop0'], layer['cache_hdrop0'] = hdrop, cache_hdrop


    for i in range(self.L):
        indx = i + 1
        w = self.params['W' + repr(indx)]
        b = self.params['b' + repr(indx)]
        h = layer['h' + repr(i)]
        if self.use_dropout:
            h = layer['hdrop'+repr(i)]
        if self.use_batchnorm and indx != (self.L):
            gamma = self.params['gamma'+repr(indx)]
            #print('gamma' + repr(indx))
            beta = self.params['beta' + repr(indx)]
            bn_param = self.bn_params['bn_param' + repr(indx)]

        # Forward Pass
        if indx == (self.L):
            h, cache_h = affine_forward(h, w, b)
            layer['h' + repr(indx)] = h
            layer['cache_h'+repr(indx)] = cache_h
        else:
            if self.use_batchnorm:
                h, cache_h = affine_norm_relu_forward(h, w, b, gamma, beta, bn_param)
                layer['h' + repr(indx)] = h
                layer['cache_h' + repr(indx)] = cache_h

            else:
                h, cache_h = affine_relu_forward(h, w, b)
                layer['h' + repr(indx)] = h
                layer['cache_h' + repr(indx)] = cache_h

            if self.use_dropout:
                h = layer['h' + str(indx)]
                hdrop, cache_hdrop = dropout_forward(h, self.dropout_param)
                layer['hdrop' + repr(indx)] = hdrop
                layer['cache_hdrop' + repr(indx)] = cache_hdrop


    scores = layer['h' + str(self.L)]
    #scores, cache_scores = affine_forward(layer[indx],
    #                                        self.params[WLast],
    #                                        self.params[bLast]
    #                                        )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y) # Calculating softmax loss

    # adding regularization loss to the data loss ( total = data loss + reg loss)::
    for i in xrange(1, self.num_layers + 1):
        loss += 0.5*self.reg*np.sum(self.params['W%d' % i]**2)

    # Begin Backpropagation
    # dx = {}
    #print("begin backprop")
    #dh[self.num_layers], grads[WLast], grads[bLast] = affine_backward(dscores, cache_scores)
    #grads[WLast] += self.reg*self.params[WLast]
    #print(self.L)
    layer['dh' + repr(self.L)] = dscores
    for i in range(self.L)[::-1]:
        indx = i + 1
        dh = layer['dh' + repr(indx)]
        h_cache =  layer['cache_h' + repr(indx)]
        if indx == self.L:
            dh, dw, db = affine_backward(dh, h_cache)
            layer['dh' + repr(indx-1)] = dh
            layer['dW' + repr(indx)] = dw
            layer['db' + repr(indx)] = db

        else:
            if self.use_dropout:
                cache_hdrop = layer['cache_hdrop' + repr(indx)]
                dh = dropout_backward(dh, cache_hdrop)
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(dh, h_cache)

                layer['dh' + repr(indx-1)] = dh
                layer['dW' + repr(indx)] = dw
                #print('dW' + repr(indx))
                layer['db' + repr(indx)] = db
                layer['dgamma' + repr(indx)] = dgamma
                #print('dgamma' + repr(indx))
                layer['dbeta' + repr(indx)] = dbeta
            else:
                dh, dw, db = affine_relu_backward(dh, h_cache)
                layer['dh' + repr(indx - 1)] = dh
                layer['dW' + repr(indx)] = dw
                layer['db' + repr(indx)] = db

    result_dw = {key[1:]:val + self.reg * self.params[key[1:]] for key, val in layer.iteritems() if key[:2] == 'dW'}
    result_db = {key[1:]:val + self.reg * self.params[key[1:]] for key, val in layer.iteritems() if key[:2] == 'db'}
    result_dgamma = {key[1:]:val + self.reg * self.params[key[1:]] for key, val in layer.iteritems() if key[:6] == 'dgamma'}
    result_dbeta = {key[1:]:val + self.reg * self.params[key[1:]] for key, val in layer.iteritems() if key[:5] == 'dbeta'}

    grads.update(result_dw)
    grads.update(result_db)
    grads.update(result_dgamma)
    grads.update(result_dbeta)

    # Modular Implementation of Backprop
    # Commented out to finish part 2 and part 3 of hw
    #for i in reversed(xrange(1, self.num_layers)):
    #    dx[i], grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dx[i + 1], cache_layer[i])
    #    grads['W%d' % i] += self.reg*self.params['W%d' % i]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    """Helper Function """
    h, h_cache = affine_forward(x, w, b)
    h_norm, h_norm_cache = batchnorm_forward(h, gamma, beta, bn_param)
    h_norm_relu, relu_cache = relu_forward(h_norm)
    cache = (h_cache, h_norm_cache, relu_cache)
    return(h_norm_relu, cache)

def affine_norm_relu_backward(dout, cache):
    """ Helper Function """
    h_cache, h_norm_cache, relu_cache = cache

    dh_norm_relu = relu_backward(dout, relu_cache)
    dh_norm, dgamma, dbeta = batchnorm_backward_alt(dh_norm_relu, h_norm_cache)
    dx, dw, db = affine_backward(dh_norm, h_cache)
    return(dx, dw, db, dgamma, dbeta)
