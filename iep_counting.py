from fast_rcnn.config import cfg
import caffe
import numpy as np
import copy
import yaml
if cfg.TEST.BOX_IEP:
    import pickle
DEBUG = False

class IepLayer(caffe.Layer):
  def setup(self, bottom, top):
    #This method is called once when caffe builds the net. This 
    #function should check that number of inputs (len(bottom)) 
    #and number of outputs (len(top)) is as expected
    # This is to initialize your weight layer

    # First read the parameter string --- i.e. the size


    layer_params = yaml.load(self.param_str)
    self.dims = int(layer_params['num_dims'])
    self.alpha = float(layer_params['alpha'])

    if cfg.TRAIN.OVERLAP and cfg.TEST.OVERLAP: 
	self.dims = 1

    self.num_classes = bottom[0].data.shape[1]
    self.num_levels = int(np.amax(bottom[1].data[:,1,0,0],axis=0))+1
    if cfg.TRAIN.FULL_IMAGE and cfg.TEST.FULL_IMAGE: 
   	self.num_levels = 1
    # Define the weights blob to be trained.	
    self.blobs.add_blob(self.num_classes, self.dims * self.dims)
    self.blobs[0].data[...] = np.random.randn(self.num_classes, self.dims*self.dims) # np.random.rand(self.num_classes, self.dims*self.dims)
    #print '[IEP] Value with which we initialize blobs:', self.blobs[0].data

  def reshape(self, bottom, top):
    #This method is called whenever caffe reshapes the net. This 
    #function should allocate the outputs (each of the top blobs). 
    #The outputs' shape is usually related to the bottoms' shape.
    top[0].reshape(self.num_levels, self.num_classes)	

  def forward(self, bottom, top):
    # bottom[0] -- box features [#batch, #class #dims], bottom[1] - functions
    if cfg.TRAIN.OVERLAP and cfg.TEST.OVERLAP: 
        patches = bottom[2].data[:,:,:,0]
    else:
        patches = bottom[0].data
        patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2] * patches.shape[3]) 	

    #print '[IEP] patches forward', patches[1,0,:], patches.shape
    level_functions = bottom[1].data
    #raw_input()

    # Define the IEP function over the weights and patches using the functions.	
    iep, class_boxpred = get_ieps_forward(self.blobs[0].data, patches, level_functions, self.num_classes)

    self.boxpred = class_boxpred	

    if cfg.TEST.BOX_IEP:
        # Write box predictions and iep(box) to file
        rois = bottom[3].data[:,:,0,0]
        max_box_preds = get_max_preds(level_functions, patches, rois, self.num_classes, self.blobs[0].data)
        with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/%s.pickle'%(image_nr),'wb') as handle:
            pickle.dump(max_box_preds,handle)


    # The loss should give back 4 numbers (1 per level and each gets multiplied with its own loss).
    top[0].reshape(*(iep.shape))
    # Copy data into net's input blobs
    top[0].data[...] = iep.astype(np.float32, copy=False)

    #Implementing the forward pass from bottom to top
    if DEBUG:
       print '[IEP] Forward pass: w -- ', self.blobs[0].data

  def backward(self, top, propagate_down, bottom):
    #This method implements the backpropagation, it propagates the 
    #gradients from top to bottom. propagate_down is a Boolean vector 
    #of len(bottom) indicating to which of the bottoms the gradient 
    #should be propagated

    # bottom[0] -- box features, bottom[1] - functions
    # Read the input bottoms: 
    if cfg.TRAIN.OVERLAP and cfg.TEST.OVERLAP: 
        patches = bottom[2].data[:,:,:,0]
    else:	
        patches = bottom[0].data.reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], self.dims*self.dims)
    level_functions = bottom[1].data

    # Get only the features from the levels not the w:
    perlevel_perclass_loss = top[0].diff # [4,21]
    dw_perbox, dw = get_ieps_backward(patches, level_functions, perlevel_perclass_loss, self.blobs[0].shape, self.num_classes, self.boxpred)

    # Set the gradient to be passed down the repeated w gradient or the gradient per box?
    dw_repeat = dw.reshape(1, dw.shape[0], dw.shape[1], 1)
    dw_repeat = np.repeat(dw_repeat, patches.shape[0], axis = 0) # repeat over boxes

    # And the gradient of the current blob is just dw
    self.blobs[0].diff[...] = dw
    if DEBUG:
       print '[IEP] Backward pass: dw -- ', self.blobs[0].diff
	

def get_ieps_forward(w, patches, level_functions, num_classes):
    levels = int(np.amax(level_functions[:,1,0,0],axis=0))+1
    if cfg.TRAIN.FULL_IMAGE and cfg.TEST.FULL_IMAGE: 
    	levels = 1
 
    class_boxpred = np.zeros((patches.shape[0], num_classes))
    iep = np.zeros((levels,num_classes))

    for level_index in range(levels):
        plus_boxes = np.where((level_functions[:,:,0,0]==[1,level_index]).all(axis=1))[0]
        minus_boxes = np.where((level_functions[:,:,0,0]==[-1,level_index]).all(axis=1))[0]
        level_iep = np.zeros(num_classes)
        for c in range(num_classes):
            w_class = w[c]

	    class_boxpred[plus_boxes,c] = np.dot(patches[plus_boxes,c,:],w_class)
    	    if cfg.TRAIN.CLIPPED_LOSS:
        	 class_boxpred[plus_boxes,c] = np.maximum(0.0, class_boxpred[plus_boxes,c])
				
            level_iep[c] = np.sum(class_boxpred[plus_boxes,c],axis=0)
            if len(minus_boxes)>0:
		class_boxpred[minus_boxes,c] = np.dot(patches[minus_boxes,c,:],w_class)
    	    
		if cfg.TRAIN.CLIPPED_LOSS:
        	    class_boxpred[minus_boxes,c] = np.maximum(0.0, class_boxpred[minus_boxes,c])

                level_iep[c] += np.sum(-1 * class_boxpred[minus_boxes,c],axis=0) 
        iep[level_index,:] = level_iep
    return iep, class_boxpred

def get_ieps_backward(patches, level_functions, perlevel_perclass_loss, w_shape, num_classes, class_boxpred):
    perlevel_perclass_loss = perlevel_perclass_loss.reshape(1, perlevel_perclass_loss.shape[0], perlevel_perclass_loss.shape[1], 1)
    perlevel_perclass_loss = np.repeat(perlevel_perclass_loss, patches.shape[0], axis=0) # repeat over boxes
    perlevel_perclass_loss = np.repeat(perlevel_perclass_loss, patches.shape[2], axis=3) # repeat over dimensions
    
    levels = int(np.amax(level_functions[:,1,0,0],axis=0))+1
    if cfg.TRAIN.FULL_IMAGE and cfg.TEST.FULL_IMAGE: 
   	 levels = 1
    dw_perbox = np.zeros(patches.shape)
    dw        = np.zeros(w_shape)
    for level_index in range(levels):
	# Find the positive and negative boxes
	pos_pred = []
	neg_pred = []
	if cfg.TRAIN.CLIPPED_LOSS:
           pos_pred = np.where((level_functions[:,:,0,0]==[1,level_index]).all(axis=1) & (class_boxpred>=0.0).all(axis=1))[0]
           neg_pred = np.where((level_functions[:,:,0,0]==[-1,level_index]).all(axis=1) & (class_boxpred>=0.0).all(axis=1))[0]
	else:
           pos_pred = np.where((level_functions[:,:,0,0]==[1,level_index]).all(axis=1))[0]
           neg_pred = np.where((level_functions[:,:,0,0]==[-1,level_index]).all(axis=1))[0]
	    
	# Multiply the box features with the layer loss.
        dw_perbox[pos_pred] = np.multiply(patches[pos_pred], perlevel_perclass_loss[pos_pred,level_index,:,:])
        if len(neg_pred)>0:
            dw_perbox[neg_pred] = -1.0 * np.multiply(patches[neg_pred], perlevel_perclass_loss[neg_pred,level_index,:,:])
	# Add the sum of the positive and negative boxes on the current level
        dw += np.sum(dw_perbox[pos_pred], axis=0) + np.sum(dw_perbox[neg_pred], axis=0)

    # Normalize by the number of levels
    dw = dw / float(levels)
    return dw_perbox, dw


    def get_max_preds(level_functions, patches, rois, num_classes, w):
        levels = int(np.amax(level_functions[:,1,0,0],axis=0))+1
        max_pred_of_level = np.zeros((num_classes, levels, 5))
        for level_index in range(levels):
            level_boxes = np.where((level_functions[:,:,:,0]==[:,level_index]).all(axis=1))[0]
            for c in range(num_classes):
                w_class = w[c]
                class_boxpred = np.dot(patches[level_boxes,c,:],w_class)
                max_ind = np.argmax(class_boxpred)
                max_pred_of_level[c,level_index,0:4] = rois[level_boxes[max_ind]]
                max_pred_of_level[c,level_index,4]   = class_boxpred[max_ind]

                #iep_box_predictions

        return max_pred_of_level