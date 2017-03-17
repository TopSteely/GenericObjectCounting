import Input
import numpy as np

load_dennis = Input.Input('mscoco','sheep',20)
train = load_dennis.coco_train_set.getImgIds()
sum_labels = np.zeros(80)
print len(train)
for i,img_nr in enumerate(train):
	sum_labels += load_dennis.get_all_labels(img_nr, 'train')
avg = sum_labels/len(train)

test = load_dennis.coco_val_set.getImgIds()
error0 = np.zeros(80)
error1 = np.zeros(80)
error_mean = np.zeros(80)

print error_mean

for i,img_nr in enumerate(test):
	y = load_dennis.get_all_labels(img_nr, 'test')
	error0 = np.abs(y)
	error1 = np.abs(y-1)
	error_mean = np.abs(y-avg)

print np.mean(error0)
print np.mean(error1)
print np.mean(error_mean)