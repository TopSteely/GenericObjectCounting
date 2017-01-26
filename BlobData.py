from utils import create_tree_as_extracted, surface_area_old, sort_boxes, create_tree
import numpy as np
import time

class BlobData():
	def __init__(self, load, img_nr):
		prune_tree_levels = 10
		self.boxes = load.get_coords_blob(img_nr)
		self.tree_boxes = self.boxes
		self.X = load.get_features_blob(img_nr, self.boxes)
		self.num_features = 3
		self.y = load.get_label_blob(img_nr)
		self.tree_boxes, self.X = sort_boxes(self.tree_boxes, self.X)
		self.boxes = self.tree_boxes
		#assert(self.boxes==self.tree_boxes)
		self.G, levels = create_tree(self.tree_boxes)
		#prune tree to only have levels which fully cover the image, tested
		total_size = surface_area_old(self.tree_boxes, levels[0])
		for level in levels:
			sa = surface_area_old(self.tree_boxes, levels[level])
			sa_co = sa/total_size
			print sa_co
			if sa_co != 1.0:
				self.G.remove_nodes_from(levels[level])
			else:
				nr_levels_covered = level
		levels = {k: levels[k] for k in range(0,nr_levels_covered + 1)}
		# prune levels, speedup + performance 
		levels_tmp = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
		levels_gone = {k:v for k,v in levels.iteritems() if k>=prune_tree_levels}
		self.levels = levels_tmp
		#prune tree as well, for patches training
		for trash_level in levels_gone.values():
			self.G.remove_nodes_from(trash_level)
		self.boxes = np.array(self.boxes)