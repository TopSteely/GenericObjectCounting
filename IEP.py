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
    def iep(self, Data, function, level):
        X = Data.X
        sets = Data.levels[level]
        coords = Data.boxes
        if np.all(self.w == 1):
            iep = 0
        else:
            iep = np.zeros(len(self.w))
        if len(sets) == 1:
            return np.dot(self.w,X[sets[0]]), []
        elif function != []:
            for fun in function:
                if '+' in fun[0]:
                    iep += np.dot(self.w,X[fun[1]])
                elif '-' in fun[0]:
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
                    overlaps.add_edges_from(comb)
        
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
                              iep += np.dot(self.w,X[ind])
                              function.append(['+',ind])
                         elif len(base)%2==0:
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
            
    def get_iep_levels(self, Data, functions):
        iep_levels = []
        for level in Data.levels:
            iep, function = self.iep(Data, [], level)
            iep_levels.append(iep)
        return iep_levels, functions
        