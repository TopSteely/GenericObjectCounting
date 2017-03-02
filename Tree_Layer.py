import caffe
import numpy as np
from utils import sort_boxes, create_tree, surface_area_old, get_intersection, get_set_intersection
import itertools
import networkx as nx
from collections import deque
from itertools import chain, islice

DEBUG = True

class Tree_Layer(caffe.Layer):
    """
    Outputs function and regions needed for the IEP
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._n_levels = layer_params['n_levels']
        self._mode = layer_params['mode']
        if self._mode == 'grid':
            self.coord_path = 'bla'
        elif self._mode == 'mscoco':
            self.coord_path = 'bla'
        elif self._mode == 'trancos':
            self.coord_path = 'bla'
        elif self._mode == 'pascal':
          self.coord_path =  '/var/node436/local/tstahl/Coords_prop_windows/%s.txt'
          self.intersection_coords_path = '/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'

        if DEBUG:
            print '#levels: {}'.format(self._n_levels)
            print 'mode: {}'.format(self._mode)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)


    def forward(self, bottom, top):
        # Algorithm:
        #
        # create tree
        # prune tree
        # append necessary intersections
        # delete needless rois

        # bottom:
        # im_info = bottom[0]

        # top[0] = patches
        # top[1] = level_functions

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        img_nr = im_info[0].data

        if DEBUG:
            print 'im_number: ({})'.format(img_nr)


        tree_boxes = self.get_coords(img_nr)
        tree_boxes,_ = sort_boxes(tree_boxes, [])
        G, levels = create_tree(tree_boxes)

        #prune tree to only have levels which fully cover the image, tested
        total_size = surface_area_old(tree_boxes, levels[0])
        for level in levels:
            sa = surface_area_old(tree_boxes, levels[level])
            sa_co = sa/total_size
            if sa_co != 1.0:
                G.remove_nodes_from(levels[level])
            else:
                nr_levels_covered = level
        levels = {k: levels[k] for k in range(100 + 1)}

        # prune levels, speedup + performance 
        levels_tmp = {k:v for k,v in levels.iteritems() if k<self._n_levels}
        levels_gone = {k:v for k,v in levels.iteritems() if k>=self._n_levels}
        levels = levels_tmp
        #prune tree as well, for patches training
        for trash_level in levels_gone.values():
            G.remove_nodes_from(trash_level)

        intersection_coords = self.get_intersection_coords(img_nr)
        #some images don't have any intersections because their tree only consist of one level
        if len(intersection_coords) > 0:
          tree_boxes = np.append(tree_boxes, intersection_coords, axis=0)
        else:
          tree_boxes = np.array(tree_boxes)

        #get level_function
        level_functions = []
        for i_level in range(len(levels)):
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
                     I = get_intersection(tree_boxes[c], I)
              if I != [] and I[1] != I[3] and I[0]!=I[2]:
                    if I in tree_boxes.tolist():
                       ind = tree_boxes.tolist().index(I)
                       if len(base)%2==1:
                          function.append(['+',ind])
                       elif len(base)%2==0:
                          function.append(['-',ind])
                    else:
                       print 'IEP: intersection not found', I
                       exit()
              for i, u in enumerate(cnbrs):
                  # Use generators to reduce memory consumption.
                  queue.append((chain(base, [u]),
                                filter(nbrs[u].__contains__,
                                       islice(cnbrs, i + 1, None))))
        level_functions[i_level] = function

        # prepend index and only keep boxes needed
        boxes_needed = []
        for function in level_functions:
          boxes_needed.extend([f[1] for f in function])
        boxes_needed = np.unique(boxes_needed)

        forward = []
        for i_box, box in enumerate(tree_boxes):
          if box in boxes_needed:
            forward.append([i_box] + box)

        forward = np.hstack((np.zeros((len(tree_boxes),1)),tree_boxes))
        
        top[0].data[...] = forward
        top[1].data[...] = level_functions

        if DEBUG:
          print forward
          print level_functions

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def get_coords(self, img_nr):
        if os.path.isfile(self.coord_path%(format(img_nr, "06d"))):
            ret = np.loadtxt(self.coord_path%(format(img_nr, "06d")), delimiter=',')
            if isinstance(ret[0], np.float64):
                return np.array([ret])
            else:
                return ret

    def get_intersection_coords(self, img_nr):
      if os.path.isfile(self.intersection_coords_path%(format(img_nr, "06d"))):
            ret = np.loadtxt(self.intersection_coords_path%(format(img_nr, "06d")), delimiter=',')
            if isinstance(ret[0], np.float64):
                return np.array([ret])
            else:
                return ret