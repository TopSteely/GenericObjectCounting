import caffe
#https://deepsense.io/region-of-interest-pooling-explained/
#i think fast and faster r-cnn make no sense
DEBUG = True

class IepLayer(caffe.Layer):
  #http://stackoverflow.com/questions/41344168/what-is-a-python-layer-in-caffe/41481539#41481539
  def setup(self, bottom, top):
    #This method is called once when caffe builds the net. This 
    #function should check that number of inputs (len(bottom)) 
    #and number of outputs (len(top)) is as expected
    # This is to initialize your weight layer
    self.blobs.add_blob(1)  
    # This is how to initialize with array
    self.blobs[0].data[...] = np.zeros(input_size,20)

  def reshape(self, bottom, top):
    #This method is called whenever caffe reshapes the net. This 
    #function should allocate the outputs (each of the top blobs). 
    #The outputs' shape is usually related to the bottoms' shape.
    pass

  def forward(self, bottom, top):
    # Algorithm:
    #
    # for each level:
      # 

    # bottom:
    # patches = bottom[0]
    # level_functions = bottom[1]
    # label = bottom[2]
    iep = np.zeros(20) #one level iep for every class
    patches_predictions = bottom[1].data
    labels = bottom[0].data # shape 20 x 1
    print patches_predictions[0]
    print labels
    raw_input()
    for level_funtion in level_functions:
      level_iep = np.zeros(20) #one level iep for every class
      for term in level_funtion:
        if term[0] == '+':
          level_iep += np.dot(self.blobs[0].data,patches_predictions[term[0]])
        else:
          level_iep -= np.dot(self.blobs[0].data,patches_predictions[term[0]])
      iep += level_iep
    top[0].data[...] = np.mean(iep - labels)

  	#Implementing the forward pass from bottom to top
    if DEBUG:
      print bottom[0].data[0]
      print bottom[1].data[0]
      print top[0].data

  def backward(self, top, propagate_down, bottom):
    #This method implements the backpropagation, it propagates the 
    #gradients from top to bottom. propagate_down is a Boolean vector 
    #of len(bottom) indicating to which of the bottoms the gradient 
    #should be propagated
    if len(set)%2==0:
      self.diff += bottom
    else:
      self.diff += bottom
    pass
