from utils import create_tree_as_extracted, surface_area_old, sort_boxes, create_tree
import numpy as np
import time

class Data:
    def __init__(self, load, img_nr, prune_tree_levels, scaler, num_features=4096, overlap_gt=False):
	#print img_nr
        self.img_nr = img_nr
        self.boxes = load.get_coords(img_nr)
        self.X = load.get_features(img_nr)
        self.num_features = num_features
        if overlap_gt:
            gr = load.get_gts(img_nr)
        elif num_features != 4096:
            features_temp = []
            for p in self.X:
                features_temp.append(p[0:num_features])
            self.X = np.array(features_temp)
        if scaler != None:
            self.X = scaler.transform(self.X)
        self.y = load.get_label(img_nr)
        #print "load", (time.time() - start)
        start = time.time()
        self.tree_boxes = load.get_coords_tree(img_nr)
        #print len(self.tree_boxes), len(self.tree_boxes[0])
        self.tree_boxes,self.X = sort_boxes(self.tree_boxes, self.X)

        if overlap_gt:
            self.y_boxes = []
            for b_i in self.tree_boxes:
                sum_tmp = 0.0
                for g_i in gr:
                    sum_tmp += get_overlap_ratio(g_i, b_i)
                    self.y_boxes.append(sum_tmp)

        #self.G, levels = create_tree_as_extracted(self.tree_boxes)
        self.G, levels = create_tree(self.tree_boxes)
        #print "tree", (time.time() - start)
        start = time.time()
        if load.mode == 'dennis':
            self.boxes = self.tree_boxes
        #prune tree to only have levels which fully cover the image, tested
        total_size = surface_area_old(self.tree_boxes, levels[0])
        for level in levels:
            sa = surface_area_old(self.tree_boxes, levels[level])
            sa_co = sa/total_size
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

            
        self.debug_tmp = []
        for n in self.G.nodes():
            self.debug_tmp.append(self.tree_boxes[n])
        if load.mode == 'pascal':
            self.lookup_coords()
        elif load.mode == 'dennis':
            assert self.boxes==self.tree_boxes
            intersection_coords = load.get_intersection_coords(img_nr)
            intersection_features = load.get_intersection_features(img_nr)
            if scaler != None and len(intersection_features) > 0:
                intersection_features = scaler.transform(intersection_features)
            assert len(intersection_coords) == len(intersection_features)
            if len(intersection_coords) > 0:
                self.boxes = np.append(self.boxes, intersection_coords, axis=0)
                if overlap_gt:
                    for is_i in intersection_coords:
                        sum_tmp = 0.0
                        for g_i in gr:
                            sum_tmp += get_overlap_ratio(g_i, is_i)
                            self.y_boxes.append(sum_tmp)
                    self.X = np.array(self.y_boxes)
                    assert self.num_features == 1
                else:
                    self.X = np.append(self.X, intersection_features[:,0:num_features], axis=0)
            else:
                self.boxes = np.array(self.boxes)
        
    def lookup_coords(self):
        #have to change level indexes because of rearranging in extraction
        levels_corrected = {}
        for level in self.levels:
            levels_corrected[level] = []
            for idx in self.levels[level]:
                coord = self.tree_boxes[idx]
                new_idx = self.boxes.tolist().index(coord.tolist())
                levels_corrected[level].append(new_idx)
        self.levels = levels_corrected
