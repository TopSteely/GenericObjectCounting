import Input
import numpy as np

load_dennis = Input.Input('mscoco','sheep',20)
train = load_dennis.coco_train_set.getImgIds()
sum_labels = np.zeros(80)
print len(train)
for i,img_nr in enumerate(train):
	lab = load_dennis.get_all_labels(img_nr, 'train')
	sum_labels += lab[1:]
avg = sum_labels/len(train)
print sum_labels
print avg
test = load_dennis.coco_val_set.getImgIds()
error0 = np.zeros(80)
error1 = np.zeros(80)
error_mean = np.zeros(80)

for i,img_nr in enumerate(test):
	y = load_dennis.get_all_labels(img_nr, 'test')
	error0 += np.abs(y[1:])
	error1 += np.abs(y[1:]-1)
	error_mean += np.abs(y[1:]-avg)

print np.mean(error0)
print np.mean(error1)
print np.mean(error_mean)