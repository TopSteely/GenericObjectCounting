from utils import create_tree_as_extracted, surface_area_old, sort_boxes, create_tree, get_overlap_ratio
import numpy as np
import time
import IEP
import itertools
from itertools import chain, islice
import networkx as nx
from collections import deque
from utils import get_set_intersection, get_intersection

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
                    self.X = np.array(self.y_boxes) + np.random.normal(0,.1,len(self.y_boxes))
                    assert self.num_features == 1
                else:
                    self.X = np.append(self.X, intersection_features[:,0:num_features], axis=0)
            else:
                if overlap_gt:
                    self.X = np.array(self.y_boxes) + np.random.normal(0,.1,len(self.y_boxes))
                    assert self.num_features == 1
                else:
                    self.boxes = np.array(self.boxes)

        print 'starting getting gt data'
        #this is just for create_mats.py
        learner = IEP.IEP(1, 'learning')
        _,function = learner.get_iep_levels(self, {})
        flevels = []
        for f in range(len(function)):
            flevels.append([a[1] for a in function[f]])
        self.box_levels = []
        temp = []
        for i in range(len(self.boxes)):
            found = False
            for i_l,fl in enumerate(flevels):
                if i in fl:
                    if found:
                        self.boxes = np.concatenate((self.boxes,self.boxes[i].reshape(1,4)), axis=0)
                       #have to put it at the end somehow
                        if function[i_l][fl.index(i)][0] == '+':
                            temp.append([1,i_l])
                        elif function[i_l][fl.index(i)][0] == '-':
                            temp.append([-1,i_l])
                    else:
                        found = True
                        if function[i_l][fl.index(i)][0] == '+':
                            self.box_levels.append([1,i_l])
                        elif function[i_l][fl.index(i)][0] == '-':
                           self.box_levels.append([-1,i_l])
            if not found:
                self.box_levels.append([0, -1])
        self.box_levels.extend(temp)
        self.level_functions = get_level_functions(self.levels,self.boxes, prune_tree_levels)

        print load.get_all_labels(self.img_nr)

        gts = load.get_all_gts(self.img_nr)
        self.gt_overlaps = np.zeros((len(self.boxes),20))
        for i_b,b in enumerate(self.boxes):
            for i_cls,cls_ in enumerate(gts):
                overlap_cls = 0.0
                for gt in gts[cls_]:
                    overlap_cls += get_overlap_ratio(gt, b)
                self.gt_overlaps[i_b,i_cls] = overlap_cls
            print self.gt_overlaps[i_b,:]
            raw_input()

        
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

def get_level_functions(levels,coords, tree_size):
  #get level_function
  # format for each level: 1 row plus patches, one row - patches, i hope 100 terms is enough
  level_functions = np.zeros((tree_size * 2, 1000))
  for i_level in range(len(levels)):
    counter_plus = 0
    counter_minus = 0
    sets = levels[i_level]
    level_coords = []
    for i in levels[i_level]:
          level_coords.append(coords[i])
    combinations = list(itertools.combinations(sets, 2))
    overlaps = nx.Graph()
    for comb in combinations:
          set_ = []
          for c in comb:
              set_.append(coords[c])
          I = get_set_intersection(set_)
          if I != []:
              overlaps.add_edges_from([comb])
    index = {}
    nbrs = {}
    for u in overlaps:
        index[u] = len(index)
        # Neighbors of u that appear after u in the iteration order of G.
        nbrs[u] = {v for v in overlaps[u] if v not in index}


    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in overlaps)
    # Loop invariants:
    # 1. len(base) is nondecreasing.
    # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
    # 3. cnbrs is a set of common neighbors of nodes in base.
    while queue:
        base, cnbrs = map(list, queue.popleft())
        I = [0,0,1000,1000]
        for c in base:
            if I != []:
               I = get_intersection(coords[c], I)
        if I != [] and I[1] != I[3] and I[0]!=I[2]:
              if I in np.array(coords):
                 ind = [list(c) for c in coords].index(I)
                 if len(base)%2==1:
                    level_functions[2*i_level,counter_plus] = ind
                    counter_plus += 1
                    if counter_plus > 999:
                        print 'more than 1000 terms'
                        exit()
                 elif len(base)%2==0:
                    level_functions[2*i_level+1,counter_minus] = ind
                    counter_minus += 1
                    if counter_minus > 999:
                        print 'more than 1000 terms'
                        exit()

              else:
                 print 'IEP: intersection not found', I
                 exit()
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None))))
  return level_functions                