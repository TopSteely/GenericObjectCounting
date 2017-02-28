import caffe


class IEP_Loss_Layer(caffe.Layer):
	#http://stackoverflow.com/questions/41344168/what-is-a-python-layer-in-caffe/41481539#41481539
  def setup(self, bottom, top):
  	#This method is called once when caffe builds the net. This 
 	#function should check that number of inputs (len(bottom)) 
 	#and number of outputs (len(top)) is as expected
    pass

  def reshape(self, bottom, top):
  	#This method is called whenever caffe reshapes the net. This 
  	#function should allocate the outputs (each of the top blobs). 
  	#The outputs' shape is usually related to the bottoms' shape.
    pass

  def forward(self, bottom, top):
  	#Implementing the forward pass from bottom to top
    pass

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