from utils import create_tree_as_extracted, surface_area_old, sort_boxes, create_tree
import numpy as np
import time
import random

class BlobData():
	def __init__(self, load, img_nr):
		prune_tree_levels = 10
		boxes = load.get_coords_blob(img_nr)
		self.boxes = self.random_bbox(boxes[0][2], boxes[0][3])
		self.tree_boxes = self.boxes
		self.X = load.get_features_blob(img_nr, self.boxes)
		self.num_features = 3
		self.y = load.get_label_blob(img_nr)
		self.tree_boxes, self.X = sort_boxes(self.tree_boxes, self.X)
		self.boxes = self.tree_boxes
		#assert(self.boxes==self.tree_boxes)
		self.G, levels = create_tree(self.tree_boxes)
		#prune tree to only have levels which fully cover the image, tested
		print levels
		total_size = surface_area_old(self.tree_boxes, levels[0])
		for level in levels:
			sa = surface_area_old(self.tree_boxes, levels[level])
			sa_co = sa/total_size
			print level, sa_co
			if sa_co != 1.0:
				self.G.remove_nodes_from(levels[level])
			else:
				nr_levels_covered = level
		raw_input()
		levels = {k: levels[k] for k in range(0,nr_levels_covered + 1)}
		# prune levels, speedup + performance 
		levels_tmp = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
		levels_gone = {k:v for k,v in levels.iteritems() if k>=prune_tree_levels}
		self.levels = levels_tmp
		print self.levels
		#prune tree as well, for patches training
		for trash_level in levels_gone.values():
			self.G.remove_nodes_from(trash_level)
		self.boxes = np.array(self.boxes)

	def random_bbox(self,im_w, im_h):
		boxes = [[0,0,im_w,im_h]]
		for b_i in range(150):
			box = []
			box.append(random.randint(0, im_w))
			box.append(random.randint(0, im_h))
			box.append(random.randint(box[0], im_w))
			box.append(random.randint(box[1], im_h))
			boxes.append(box)
		#make sure some are at the corners
		for b_i in range(10):
			box = []
			box.append(0)
			box.append(random.randint(0, im_h))
			box.append(random.randint(box[0], im_w))
			box.append(random.randint(box[1], im_h))
			boxes.append(box)
		for b_i in range(10):
			box = []
			box.append(random.randint(0, im_w))
			box.append(0)
			box.append(random.randint(box[0], im_w))
			box.append(random.randint(box[1], im_h))
			boxes.append(box)
		for b_i in range(10):
			box = []
			box.append(random.randint(0, im_w))
			box.append(random.randint(0, im_h))
			box.append(im_w)
			box.append(random.randint(box[1], im_h))
			boxes.append(box)
		for b_i in range(10):
			box = []
			box.append(random.randint(0, im_w))
			box.append(random.randint(0, im_h))
			box.append(random.randint(box[0], im_w))
			box.append(im_h)
			boxes.append(box)
		return boxes