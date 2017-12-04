import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.misc import imread

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
            intersections = 0
            while queue:
                base, cnbrs = map(list, queue.popleft())
                if len(base) > length:
                    length = len(base)
                I = [0,0,3000,3000]
                for c in base:
                    if I != []:
                       I = get_intersection(coords[c], I)
                if I != [] and I[1] != I[3] and I[0]!=I[2]:
                      if I in coords.tolist():
                        ind = coords.tolist().index(I)


                        # plot images as nodes
                        if False:
                            if level > 2 and len(base) > 4:
                                img=imread('/var/node436/local/tstahl/Images/%s.jpg'%(format(Data.img_nr, "06d")))
                                pos=nx.circular_layout(overlaps)
                                edges = overlaps.edges()
                                #print base, u in base, v in base, u in base and v in base
                                colors = ['y' if (u in base and v in base) else 'b' for u,v in edges]
                                nx.draw(overlaps,pos,edge_color=colors)

                                # add images on edges
                                ax=plt.gca()
                                fig=plt.gcf()
                                trans = ax.transData.transform
                                trans2 = fig.transFigure.inverted().transform
                                imsize = 0.2 # this is the image size
                                for n in overlaps.nodes():
                                    (x,y) = pos[n]
                                    xx,yy = trans((x,y)) # figure coordinates
                                    xa,ya = trans2((xx,yy)) # axes coordinates
                                    img_node =  img[coords[n][1]:coords[n][3], coords[n][0]:coords[n][2]]
                                    a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
                                    a.imshow(img_node)
                                    a.set_aspect('equal')
                                    a.axis('off')
                                

                                img_inter = img[coords[ind][1]:coords[ind][3], coords[ind][0]:coords[ind][2]]
                                newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE', zorder=-1)
                                newax.imshow(img_inter)
                                newax.axis('off')


                                plt.savefig('/var/node436/local/tstahl/graph_%s_%s_%s.pdf'%(Data.img_nr, level, intersections)) 
                                intersections += 1
                                plt.clf()
                                print 'saved ', Data.img_nr, level

#-------------------------------------------------
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
            print 'levels: ', level
            if level in functions:
                # if clip there might be all predictions < 0 and the derivative is just y
                if functions[level] == []:
                    iep = np.zeros(len(Data.X[0]))
                else:
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
                I = [0,0,3000,3000]
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
        
