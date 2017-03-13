from fast_rcnn.config import cfg
import caffe
import numpy as np
import copy
DEBUG = False

class LossLayer(caffe.Layer):
  def setup(self, bottom, top):
    #This method is called once when caffe builds the net. This 
    #function should check that number of inputs (len(bottom)) 
    #and number of outputs (len(top)) is as expected
    # This is to initialize your weight layer
    pass

  def reshape(self, bottom, top):
    #This method is called whenever caffe reshapes the net. This 
    #function should allocate the outputs (each of the top blobs). 
    #The outputs' shape is usually related to the bottoms' shape.
    top[0].reshape(*bottom[0].data.shape)	

  def forward(self, bottom, top):
    # Read the bottom info: 	
    labels = bottom[0].data # shape 21 x 1
    iep = bottom[1].data

    	
    # Repeat the labels per level (we have the same label for all levels, but not the same predictions)
    repeat_labels = labels[0,:,0,0]
    repeat_labels = repeat_labels.reshape(1, labels.shape[1])

    # Repeat the labels over levels 
    if cfg.TRAIN.FULL_IMAGE==False or cfg.TEST.FULL_IMAGE==False: 
    	repeat_labels = np.repeat(repeat_labels, iep.shape[0], axis=0)		 

    self.diff = (iep - repeat_labels)

    # Prepare the top to be sent up.	
    top[0].reshape(*(self.diff.shape))

    # Copy data into net's input blobs
    if cfg.TRAIN.FULL_IMAGE or cfg.TEST.FULL_IMAGE: 
    	top[0].data[...] = np.abs(self.diff.astype(np.float32, copy=False))
    else:	
    	top[0].data[...] = np.mean(np.abs(self.diff.astype(np.float32, copy=False)), axis=0)

    top[0].data[...] = top[0].data

    self.loss = top[0].data
    #Implementing the forward pass from bottom to top
    if DEBUG:
       print '[LOSS] Froward pass: self.diff - ', self.diff

  def backward(self, top, propagate_down, bottom):
    #This method implements the backpropagation, it propagates the 
    #gradients from top to bottom. propagate_down is a Boolean vector 
    #of len(bottom) indicating to which of the bottoms the gradient 
    #should be propagate
    bottom[1].diff[...] = np.sign(self.diff)
    
    if DEBUG:
       print '[LOSS] Backward pass: bottom[1].diff - ', bottom[1].diff

"""
def get_ieps(patches_predictions,level_functions,):
    iep = np.zeros((len(level_functions)/2,21)) #one level iep for every class
    for level_index in range(len(level_functions)/2):
      if level_index == 0:
          level_iep = copy.deepcopy(patches_predictions[0])
          patches_predictions[0] = np.zeros(patches_predictions.shape[1])
      else:
          level_iep = np.sum(patches_predictions[level_functions[2*level_index,:,0,0].astype('int32')],axis=0)
          level_iep -= np.sum(patches_predictions[level_functions[2*level_index+1,:,0,0].astype('int32')],axis=0)

      iep[level_index,:] = level_iep
    return iep

def get_ieps_(patches_predictions,level_functions):
    levels = int(np.amax(level_functions[:,1,0,0],axis=0))
    iep = np.zeros((levels,21))
    for level_index in range(levels):
        plus_boxes = np.where((level_functions[:,:,0,0]==[1,level_index]).all(axis=1))[0]
        minus_boxes = np.where((level_functions[:,:,0,0]==[-1,level_index]).all(axis=1))[0]
        if len(plus_boxes) == 1:
           level_iep = np.sum(patches_predictions[plus_boxes],axis=0)
        else:
            level_iep = np.sum(patches_predictions[plus_boxes],axis=0)
        if len(minus_boxes)>0:
            level_iep = np.sum( -1. * patches_predictions[minus_boxes],axis=0)
        iep[level_index,:] = level_iep
    return iep
"""
