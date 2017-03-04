import caffe
#https://deepsense.io/region-of-interest-pooling-explained/
#i think fast and faster r-cnn make no sense
DEBUG = True

class IEP_Loss_Layer(caffe.Layer):
  #http://stackoverflow.com/questions/41344168/what-is-a-python-layer-in-caffe/41481539#41481539
  def setup(self, bottom, top):
  #This method is called once when caffe builds the net. This 
  #function should check that number of inputs (len(bottom)) 
  #and number of outputs (len(top)) is as expected

  def reshape(self, bottom, top):
    #This method is called whenever caffe reshapes the net. This 
    #function should allocate the outputs (each of the top blobs). 
    #The outputs' shape is usually related to the bottoms' shape.
    pass

  def forward(self, bottom, top):
    # bottom:
    # patches = bottom[0]
    # level_functions = bottom[1]
    # label = bottom[2]
    score = np.zeros(20) #one level iep for every class
    predictions = bottom[0].data
    level_functions = bottom[1].data
    labels = bottom[2].data # shape 20 x 1
    for level_funtion in level_functions:
      level_score = np.zeros(20) #one level iep for every class
      for term in level_funtion:
        if term[0] == '+':
          level_score += predictions[term[1]]
        elif term[0] == '-':
          level_score -= predictions[term[1]]
      score += level_score
    top[0].data[...] = np.mean(score - labels)

  def backward(self, top, propagate_down, bottom):
    #This method implements the backpropagation, it propagates the 
    #gradients from top to bottom. propagate_down is a Boolean vector 
    #of len(bottom) indicating to which of the bottoms the gradient 
    #should be propagated
    diff = np.zeros(feature_size)
    labels = bottom[2].data
    for level_funtion in level_functions:
      level_diff = np.zeros(feature_size)
      for term in level_funtion:
        if term[0] == '+':
          level_diff += bottom[0].data[term[1]]
        else:
          level_diff -= bottom[0].data[term[1]]
      diff += np.sign(top[0].diff) * level_diff
    bottom[0].diff[...] = np.mean(diff)
