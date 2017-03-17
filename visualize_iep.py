import matplotlib
matplotlib.use('agg')
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.misc import imread
from matplotlib.patches import Rectangle

with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/max_box_pred248.0.pickle','rb') as handle:
	max_box = pickle.load(handle)

with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/max_iep_box_pred248.0.pickle','rb') as handle:
	iep_box = pickle.load(handle)

with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/iep248.0.pickle','rb') as handle:
	iep = pickle.load(handle)

print max_box.shape, iep_box.shape,iep.shape

im = imread('/var/node436/local/tstahl/Images/000248.jpg')
for lvl in range(iep.shape[0]):
	for i_c, class_ in enumerate(['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']):
#		pp = PdfPages('/var/node436/local/tstahl/vis_iep/%s_%s_%s.pdf'%(248,lvl,class_))
		plt.imshow(im)
		plt.axis('off')
		ax = plt.gca()
		coord_iep = max_box[i_c,lvl,0:4]
		coord_iep_ = iep_box[i_c,lvl,0:4]
		lvl_pred = iep[lvl,i_c]

		box_ = max_box[i_c,lvl,4]
		iep_box_ = iep_box[i_c,lvl,4]

		ax.add_patch(Rectangle((int(coord_iep[0]), int(coord_iep[1])), int(coord_iep[2] - coord_iep[0]), int(coord_iep[3] - coord_iep[1]), edgecolor='red', facecolor='none'))
		ax.add_patch(Rectangle((int(coord_iep_[0]), int(coord_iep_[1])), int(coord_iep_[2] - coord_iep_[0]), int(coord_iep_[3] - coord_iep_[1]), edgecolor='green', facecolor='none'))
		ax.set_title('(r)best Patch: %s, (g)best IEP Patch: %s\n IEP Level: %s'%(box_,iep_box_,lvl_pred))

		plt.title('Error per level')
#		pp.savefig()
		plt.savefig('/var/node436/local/tstahl/vis_iep/%s_%s_%s.pdf'%(248,lvl,class_), bbox_inches='tight')
		plt.clf()
pp.close()