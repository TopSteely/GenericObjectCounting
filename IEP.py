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
        print sets
        raw_input()
        coords = Data.boxes
        if np.all(self.w == 1):
            iep = np.zeros(Data.num_features)
        else:
            iep = 0
        if len(sets) == 1:
            #if function == []:
                #function.append(['+',sets[0]])
#            if np.all(self.w == 1):
#                print 'root: ', sets[0], (X[sets[0]]==0).sum(), self.w
#            else:
#                print 'root: ', sets[0], (X[sets[0]]==0).sum(), self.w.sum(), len(self.w)
            return np.dot(self.w,X[sets[0]]), function
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
                print base
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
                            print '+', X[ind]
                            iep += np.dot(self.w,X[ind])
                            #function.append(['+',ind])
                         elif len(base)%2==0:
                            print '-', X[ind]
                            iep -=  np.dot(self.w,X[ind])
                            #function.append(['-',ind])
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
            if level in functions:
                iep, function = self.iep(Data, functions[level], level)
            else:
                iep, function = self.iep(Data, [], level)
                #functions[level] = function
            iep_levels.append(iep)
        return iep_levels, functions
        
