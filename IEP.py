import itertools
import networkx as nx
from utils import get_set_intersection
from collections import deque
from itertools import chain, islice
from utils import get_intersection
import numpy as np

class IEP:
    def __init__(self, w, mode):
        if mode == 'prediction':
            self.w = w
        else:
            self.w = 1
        
    #returns the cardinality of the union of sets
    def iep(self, Data, function, level, clip):
        X = Data.X
        sets = Data.levels[level]
        coords = Data.boxes
        if np.all(self.w == 1):
            iep = np.zeros(Data.num_features)
        else:
            iep = 0
        if len(sets) == 1:
            if function == []:
                if clip:
                    if np.dot(self.w,X[sets[0]]) > 0:
                        function.append(['+',sets[0]])
                else:
                    function.append(['+',sets[0]])

            if clip:
                return max(0,np.dot(self.w,X[sets[0]])), function
            else:
                return np.dot(self.w,X[sets[0]]), function
        elif function != []:
            if clip:
                print 'never go here when clipped!!!!'
                exit()
            for fun in function:
                if '+' in fun[0]:
                    if clip:
                        iep += max(0,np.dot(self.w,X[fun[1]]))
                    else:
                        iep += np.dot(self.w,X[fun[1]])
                elif '-' in fun[0]:
                    if clip:
                        iep -= max(0,np.dot(self.w,X[fun[1]]))
                    else:
                        iep -= np.dot(self.w,X[fun[1]])
                else:
                    print 'wrong symbol 0', fun[0]
                    exit()
            return iep, function
        else:
            level_coords = []
            for i in sets:
                level_coords.append(coords[i])
            combinations = list(itertools.combinations(sets, 2)) 
            overlaps = nx.Graph()
            
            for comb in combinations:
                set_ = []
                for c in comb:
                    set_.append(coords[c])
                I = get_set_intersection(set_)
                if I != []:
                    #overlaps.add_edge(comb[0],comb[1])
                    overlaps.add_edges_from([comb])
        
            length = 1
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
                if len(base) > length:
                    length = len(base)
                I = [0,0,1000,1000]
                for c in base:
                    if I != []:
                       I = get_intersection(coords[c], I)
                if I != [] and I[1] != I[3] and I[0]!=I[2]:
                      if I in coords.tolist():
                         ind = coords.tolist().index(I)
                         if ind >= len(X):
                             print 'index bigger than X'
                             exit()
                         if len(base)%2==1:
                            #print '+', X[ind]
                            if clip:
                                iep += max(0,np.dot(self.w,X[ind]))
                                if np.dot(self.w,X[ind]) > 0:
                                    function.append(['+',ind])
                            else:
                                iep += np.dot(self.w,X[ind])
                                function.append(['+',ind])
                         elif len(base)%2==0:
                            #print '-', X[ind]
                            if clip:
                                iep -=  max(0,np.dot(self.w,X[ind]))
                                if np.dot(self.w,X[ind]) > 0:
                                    function.append(['-',ind])
                            else:
                                iep -=  np.dot(self.w,X[ind])
                                function.append(['-',ind])
                      else:
                         print 'IEP: intersection not found', I
                         exit()
                for i, u in enumerate(cnbrs):
                    # Use generators to reduce memory consumption.
                    queue.append((chain(base, [u]),
                                  filter(nbrs[u].__contains__,
                                         islice(cnbrs, i + 1, None))))
            return iep, function
            
    def get_iep_levels(self, Data, functions, clip=False):
        iep_levels = []
        for level in Data.levels:
            print level, level in functions, functions[level], clip
            if level in functions:
                iep, function = self.iep(Data, functions[level], level, clip)
            else:
                iep, function = self.iep(Data, [], level, clip)
                functions[level] = function
            iep_levels.append(iep)
        return iep_levels, functions


    def iep_single_patch(self, Data, function, level):
        if len(Data.levels) == 1:
            return [np.dot(self.w,Data.X[0])]
        count_per_level_temp = 0
        level_boxes = []
        for i in Data.levels[level]:
            level_boxes.append(Data.boxes[i])
        
        # create graph G from combinations possible        
        combinations = list(itertools.combinations(Data.levels[level], 2)) 
        iep_patch = []
        all_patches = [a[1] for a in function]

        for iep_node in all_patches: #before: Data.levels[level], but i want to have intersections as well
            comb_node = [its for its in combinations if iep_node in its]
            G = nx.Graph()
            G.add_edges_from(comb_node)
            for comb in comb_node:
                set_ = []
                for c in comb:
                    set_.append(Data.boxes[c])
                I = get_set_intersection(set_)
                if I == []:
                    G.remove_edges_from([comb])
            
            feat_exist = True #must be false in order to write
            length = 1
            index = {}
            nbrs = {}
        
            for u in G:
                index[u] = len(index)
                # Neighbors of u that appear after u in the iteration order of G.
                nbrs[u] = {v for v in G[u] if v not in index}
        
            queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
            # Loop invariants:
            # 1. len(base) is nondecreasing.
            # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
            # 3. cnbrs is a set of common neighbors of nodes in base.
            while queue:
                base, cnbrs = map(list, queue.popleft())
                if iep_node not in base:
                    continue
                if len(base) > length:
                    length = len(base)
                I = [0,0,1000,1000]
                for c in base:
                    if I != []:
                       I = get_intersection(Data.boxes[c], I)
                if I != []:
                  if I in Data.boxes:
                     ind = Data.boxes.tolist().index(I)
                     if len(base)%2==1:
                         count_per_level_temp += np.dot(self.w,Data.X[ind])
                     else:
                         count_per_level_temp -= np.dot(self.w,Data.X[ind])
                  else:
                    print 'not found'
                    exit()
                for i, u in enumerate(cnbrs):
                    # Use generators to reduce memory consumption.
                    queue.append((chain(base, [u]),
                                  filter(nbrs[u].__contains__,
                                         islice(cnbrs, i + 1, None))))
            iep_patch.append(count_per_level_temp)
        return iep_patch
        
