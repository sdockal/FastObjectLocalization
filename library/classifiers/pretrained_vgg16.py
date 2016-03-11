import numpy as np
import h5py

from library.layers import *
from library.fast_layers import *
from library.layer_utils import *


class PretrainedVGG(object):
  def __init__(self, dtype=np.float32, num_classes=1000, input_size=224, h5_file=None,verbose=False):
    #Input Size is 224X224
    #Num classes = 1000 for ImageNet
    
    self.dtype = dtype
    self.conv_params = []
    self.input_size = input_size
    self.num_classes = num_classes
    
    # TODO: In the future it would be nice if the architecture could be loaded from
    # the HDF5 file rather than being hardcoded. For now this will have to do.
    
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    
    #Map Describing the layer type
    self.layer_map = {1:'conv', 2:'relu', 3:'conv', 4: 'relu', 5:'maxpool',
                      6:'conv', 7:'relu', 8:'conv', 9: 'relu', 10:'maxpool',
                      11:'conv', 12:'relu', 13:'conv', 14: 'relu', 15:'conv', 16:'relu' ,17:'maxpool',
                      18:'conv', 19:'relu', 20:'conv', 21: 'relu', 22:'conv', 23:'relu' ,24:'maxpool',
                      25:'conv', 26:'relu', 27:'conv', 28: 'relu', 29:'conv', 30:'relu' ,31:'maxpool',
                      32:'affine', 33:'relu', 34:'affine', 35:'relu', 36:'affine',37:'softmax'
                     }
    #self. weight_layers = {1:1,2:3,3:6,4:8,5:11,6:13,7:15,8:18,9:20,10:22,11:25,12:27,13:29,14:32,15:34,16:36}
    self.weight_layers_inv = {1:1,3:2,6:3,8:4,11:5,13:6,15:7,18:8,20:9,22:10,25:11,27:12,29:13,32:14,34:15,36:16}

    self.filter_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    self.num_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    hidden_dim_1 = 4096
    hidden_dim_2 = 4096
    
    cur_size = input_size
    prev_dim = 3
    self.params = {}
    
    for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
      self.params['W%d' % (i + 1)] = np.random.randn(next_dim, prev_dim, f, f)
      self.params['b%d' % (i + 1)] = np.zeros(next_dim)
      prev_dim = next_dim
      
    cur_size = cur_size/32
    # Add a fully-connected layers
    fan_in = cur_size * cur_size * self.num_filters[-1]
    self.params['W%d' % (i + 2)] = np.zeros((fan_in,hidden_dim_1))
    self.params['b%d' % (i + 2)] = np.zeros(hidden_dim_1)
    
    self.params['W%d' % (i + 3)] = np.zeros((hidden_dim_1, hidden_dim_2))
    self.params['b%d' % (i + 3)] = np.zeros(hidden_dim_2)

    self.params['W%d' % (i + 4)] = np.zeros((hidden_dim_2, num_classes))
    self.params['b%d' % (i + 4)] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

    if h5_file is not None:
      self.load_weights(h5_file, verbose)
      pass

  
  def load_weights(self, h5_file, verbose=False):
    """
    Load pretrained weights from an HDF5 file.

    Inputs:
    - h5_file: Path to the HDF5 file where pretrained weights are stored.
    - verbose: Whether to print debugging info
    """
    with h5py.File(h5_file, 'r') as f:
      for k, v in f.iteritems():
        layer_no = int(k.split('_')[1]) 
        if(layer_no in self.weight_layers_inv.keys() ):
            for key,value in v.iteritems():
                wt_layer_no = self.weight_layers_inv[layer_no]
                if('param_0' in value.name):
                   param_name = 'W%d' % wt_layer_no
                   param_value = np.asarray(value.value)
                   if self.layer_map[layer_no] == 'conv':
                        NF, D,W,H = param_value.shape
                        param_value[:,:,range(W-1,-1,-1),:] =param_value.copy()
                        param_value[:,:,:,range(H-1,-1,-1)] =param_value.copy()
                elif ('param_1' in value.name):
                   param_name = 'b%d' % wt_layer_no
                   param_value = np.asarray(value.value)
                if verbose: print param_name, self.params[param_name].shape
                if param_value.shape == self.params[param_name].shape:
                    self.params[param_name] = param_value
                elif param_value.T.shape == self.params[param_name].shape:
                    asself.params[param_name] = param_value.T
                else:
                    raise ValueError('shapes for %s do not match' % param_name)
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)

  
  def forward(self, X, start=None, end=None, mode='test'):
    """
    Run part of the model forward, starting and ending at an arbitrary layer,
    in either training mode or testing mode.

    You can pass arbitrary input to the starting layer, and you will receive
    output from the ending layer and a cache object that can be used to run
    the model backward over the same set of layers.

    For the purposes of this function, a "layer" is one of the following blocks:

    [conv3-64 - relu] X 2
    MaxPool
    [conv3-128 - relu] X 2
    MaxPool
    [conv3-256 - relu] X 3
    MaxPool
    [conv3-512 - relu] X 2
    [conv1-512 - relu] X 1
    MaxPool
    [conv3-512 - relu] X 2
    [conv1-512 - relu] X 1
    MaxPool
    [affine - relu] X 2
    [affine] 
    Softmax
    
    Layer 0
    Layer1 - Conv3-64
    Layer2 - Relu
    Layer3 - Conv3-64
    Layer4 - Relu
    Layer5 - Maxpool
    
    Layer6 - Conv3-128
    Layer7 - Relu
    Layer8 - Conv3-128
        9  - Relu
        10 - Maxpool
        
        11 - Conv3-256
        12 - Relu
        13 - Conv3-256
        14 - Relu
        15 - Conv3-256
        16 - Relu
        17 - Maxpool
        
        18 - conv3-512
        19 - Relu
        20 - Conv3-512
        21 - Relu
        22 - Conv1-512
        23 - Relu
        24 - Maxpool
        
        25 - conv3-512
        26 - Relu
        27 - Conv3-512
        28 - Relu
 
        29 - Conv3-512
        30 - Relu
        31 - Maxpool
        
        32 - Affine FC-4096
        33 - Relu
        34 - Affine FC-4096
        35 - Relu
        36 - Affine FC-1000
        
        37 - Softmax
        
                
    Inputs:
    - X: The input to the starting layer. If start=0, then this should be an
      array of shape (N, C, 64, 64).
    - start: The index of the layer to start from. start=0 starts from the first
      convolutional layer. Default is 0.
    - end: The index of the layer to end at. start=11 ends at the last
      fully-connected layer, returning class scores. Default is 11.
    - mode: The mode to use, either 'test' or 'train'. We need this because
      batch normalization behaves differently at training time and test time.

    Returns:
    - out: Output from the end layer.
    - cache: A cache object that can be passed to the backward method to run the
      network backward over the same range of layers.
    """
    max_pool_layers = [2,4,7,10,13]
    X = X.astype(self.dtype)
    if start is None: start = 0
    if end is None: end = len(self.conv_params) + 2
    layer_caches = []
    pool_params = {'stride':2 , 'pool_height':2, 'pool_width':2 }
    prev_a = X
    for i in xrange(start, end + 1):
      i1 = i + 1
      if 0 <= i < len(self.conv_params):
        # This is a conv layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        conv_param = self.conv_params[i]
        if((i+1) in max_pool_layers):
            next_a, cache = conv_relu_pool_forward(prev_a, w, b, conv_param, pool_params)
        else:
            next_a, cache = conv_relu_forward(prev_a, w, b, conv_param)
      elif i < len(self.conv_params) + 2:
        # This is the fully-connected hidden layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        prev_a = prev_a.reshape((-1,w.shape[0]))
        next_a, cache = affine_relu_forward(prev_a, w, b)
        #print "Layer ", i1
        #print next_a.shape, "affine"
        #print np.mean(next_a), np.argmax(next_a)
        #print next_a[0,0:10]
      elif i == len(self.conv_params) + 2:
        # This is the last fully-connectsed layer that produces scores
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        next_a, cache = affine_forward(prev_a, w, b)
        #print "layer", i1
        #print next_a.shape, "last_affine"
        #print np.mean(next_a), np.argmax(next_a)
      else:
        raise ValueError('Invalid layer index %d' % i)

      layer_caches.append(cache)
      prev_a = next_a

    out = prev_a
    cache = (start, end, layer_caches)
    return out, cache


  def backward(self, dout, cache):
    """
    Run the model backward over a sequence of layers that were previously run
    forward using the self.forward method.

    Inputs:
    - dout: Gradient with respect to the ending layer; this should have the same
      shape as the out variable returned from the corresponding call to forward.
    - cache: A cache object returned from self.forward.

    Returns:
    - dX: Gradient with respect to the start layer. This will have the same
      shape as the input X passed to self.forward.
    - grads: Gradient of all parameters in the layers. For example if you run
      forward through two convolutional layers, then on the corresponding call
      to backward grads will contain the gradients with respect to the weights,
      biases, and spatial batchnorm parameters of those two convolutional
      layers. The grads dictionary will therefore contain a subset of the keys
      of self.params, and grads[k] and self.params[k] will have the same shape.
    """
    start, end, layer_caches = cache
    dnext_a = dout
    grads = {}
    max_pool_layers = [2,4,7,10,13]

    j = len(layer_caches) - 1
    for i in reversed(range(start, end + 1)):
      i1 = i + 1
      if i == len(self.conv_params) + 2:
        # This is the last fully-connected layer
        print i
        dprev_a, dw, db = affine_backward(dnext_a, layer_caches[j])
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
      elif i >= len(self.conv_params):
        # This is the fully-connected hidden layer
        j = len(layer_caches) - 1
        temp = affine_relu_backward(dnext_a, layer_caches[j])
        dprev_a, dw, db = temp
        if i ==  len(self.conv_params):
          dprev_a = dprev_a.reshape((dprev_a.shape[0],512,7,7))
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
      elif 0 <= i < len(self.conv_params):
        # This is a conv layer
        if((i+1) in max_pool_layers):
            temp = conv_relu_pool_backward(dnext_a,layer_caches[j] )
        else:
            temp = conv_relu_backward(dnext_a, layer_caches[j])
        dprev_a, dw, db = temp
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
      else:
        raise ValueError('Invalid layer index %d' % i)
      dnext_a = dprev_a
      j = j-1

    dX = dnext_a
    return dX, grads


  def loss(self, X, y=None):
    """
    Classification loss used to train the network.

    Inputs:
    - X: Array of data, of shape (N, 3, 64, 64)
    - y: Array of labels, of shape (N,)

    If y is None, then run a test-time forward pass and return:
    - scores: Array of shape (N, 100) giving class scores.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar giving loss
    - grads: Dictionary of gradients, with the same keys as self.params.
    """
    # Note that we implement this by just caling self.forward and self.backward
    mode = 'test' if y is None else 'train'
    scores, cache = self.forward(X, mode=mode)
    if mode == 'test':
      return scores
    loss, dscores = softmax_loss(scores, y)
    dX, grads = self.backward(dscores, cache)
    return loss, grads

