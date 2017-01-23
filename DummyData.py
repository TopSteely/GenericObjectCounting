from utils import create_tree_as_extracted, surface_area_old, sort_boxes, create_tree
import numpy as np
import time

class DummyData():
	def __init__(self):
		prune_tree_levels = 5
		self.boxes = np.array([[0,0,100,100],[0,0,50,100],[40,0,100,100],[0,0,45,100],[35,0,100,100]])
		self.tree_boxes = self.boxes
		self.X = np.array([[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]])
		self.num_features = 5
		self.y = 4
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

		#appending intersections:
		self.boxes = np.array(self.boxes)
		print self.boxes.shape
		print np.array([35, 0, 50, 100]).reshape(1,-1).shape
		self.boxes = np.append(self.boxes, [35, 0, 50, 100], axis=1)
		print self.boxes
		self.X = np.append(self.X, [1, 1, 1, 1, 1], axis=0)
		self.boxes = np.array(self.boxes)