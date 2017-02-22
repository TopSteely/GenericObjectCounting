import networkx as nx
from collections import deque
import itertools
from itertools import chain, islice
import numpy as np
from copy import deepcopy
import math


def lower_constraint(w,x,y,alpha,level_fcts):
    ret = 0.0
    for x_ in x:
        ret += np.minimum(np.dot(np.array(x_),w)-1,0).sum()
    return ret


def upper_constraint(w,x,y,alpha,level_fcts):
    ret = 0.0
    for x_,y_ in zip(x,y):
        ret += np.minimum(y_-np.dot(np.array(x_),w),0).sum()
    return ret


def loss_new_scipy(w, x, y, alpha, fct):
    loss = 0.0
    for img_nr, img_fct in zip(fct.keys(),fct.values()):
        for level_fct in img_fct:
            for fun in level_fct:
                copy = deepcopy(level_fct)
                copy.remove(fun)
                iep = iep_with_func(w,x[img_nr],copy)
                window_pred = np.dot(w, x[img_nr][fun[1]])
                if fun[0] == '+':
                    loss += ((y[img_nr] - iep - window_pred) ** 2)
                elif fun[0] == '-':
                    loss += ((y[img_nr] - iep + window_pred) ** 2)
        loss+= alpha * math.sqrt(np.dot(w,w))
    print loss, alpha * math.sqrt(np.dot(w,w))
    return loss #+ alpha * math.sqrt(np.dot(w,w))


def iep_with_func(w, X, function):
    iep = 0
    for fun in function:
        if '+' in fun[0]:
            iep += np.dot(w,X[fun[1]])
        elif '-' in fun[0]:
            iep -= np.dot(w,X[fun[1]])
        else:
            print 'wrong symbol 0', fun[0]
            exit()
    return iep


def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3])
    

def get_overlap_ratio(A, B):
    in_ = bool_rect_intersect(A, B)
    if not in_:
        return 0
    else:
        left = max(A[0], B[0]);
        top = max(A[1], B[1]);
        right = min(A[2], B[2]);
        bottom = min(A[3], B[3]);
        intersection = [left, top, right, bottom];
        surface_intersection = (intersection[2]-intersection[0])*(intersection[3]-intersection[1]);
        surface_A = (A[2]- A[0])*(A[3]-A[1]) + 0.0;
        return surface_intersection / surface_A
    
def get_intersection(A, B):
    in_ = bool_rect_intersect(A, B)
    if not in_:
        return []
    else:
        left = max(A[0], B[0]);
        right = min(A[2], B[2]);
        top = max(A[1], B[1]);
        bottom = min(A[3], B[3]);
        intersection = [left, top, right, bottom];
        return intersection
        
        
def create_tree(boxes):
    G = nx.Graph()
    levels = {}
    levels[0] = [0]
    G.add_node(0)
    if len(boxes) != 1:
        for box, i in zip(boxes[1:len(boxes)], range(1,len(boxes))):
            if (box[2]-box[0]) * (box[3]-box[1]) == 0: # some boxes have a surface area of 0 like (0,76,100,76)
                #print box
                #print 'surface area of box == 0', i
                continue
            possible_parents = []
            for box_, ii in zip(boxes, range(len(boxes))):
                if get_overlap_ratio(box, box_) == 1 and np.any(box != box_):
                    possible_parents.append(ii)
                    #print i, '-', ii
            I = boxes[i]
            put_here = []
            for pp in possible_parents:
                p_h = True
                if nx.has_path(G,pp,pp):
                    level = nx.shortest_path_length(G,0,pp)+1
                    if level in levels:
                        for window in levels[level]:
                            II = boxes[window]
                            if get_overlap_ratio(I, II) == 1:
                                p_h = False
                        if p_h == True:
                            put_here.append(pp)
                    else:
                        put_here.append(pp)
                else:
                    print 'do we ever go here?'
                    put_here.append(pp)
            parent = min(put_here)
            G.add_edge(i,parent)
            level = nx.shortest_path_length(G,0,parent)+1
            if level in levels:
                if parent not in levels[level]:
                    levels[level].append(i)
            else:
                levels[level] = [i]

    return G, levels
    
    
def create_tree_as_extracted(boxes):
    G = nx.Graph()
    levels = {}
    levels[0] = [0]
    G.add_node(0)
    if len(boxes) != 1:
        for box, i in zip(boxes[1:len(boxes)], range(1,len(boxes))):
            if (box[2]-box[0]) * (box[3]-box[1]) == 0: # some boxes have a surface area of 0 like (0,76,100,76)
                continue
            possible_parents = []
            for box_, ii in zip(boxes, range(len(boxes))):
		#print get_overlap_ratio(box, box_), box != box_
                if get_overlap_ratio(box, box_) == 1 and np.array_equal(box,box_): # was box != box_
                    possible_parents.append(ii)
                    #print i, '-', ii
            I = boxes[i]
            put_here = []
            for pp in possible_parents:
                p_h = True
                level = nx.shortest_path_length(G,0,pp)+1
                if level in levels:
                    for window in levels[level]:
                        II = boxes[window]
                        if get_overlap_ratio(I, II) == 1:
                            p_h = False
                    if p_h == True:
                        put_here.append(pp)
                else:
                    put_here.append(pp)
            parent = min(put_here)
            level = nx.shortest_path_length(G,0,parent)+1
            if level in levels:
                if parent not in levels[level]:
                    levels[level].append(i)
                G.add_edge(i,parent)
            else:
                levels[level] = [i]
                G.add_edge(i,parent)

    return G, levels
    
    
def create_tree_old(boxes):
    G = nx.Graph()
    levels = {}
    levels[0] = [0]
    G.add_node(0)
    print len(boxes)
    if len(boxes) != 1:
        for box, i in zip(boxes[1:len(boxes)], range(1,len(boxes))):
            if (box[2]-box[0]) * (box[3]-box[1]) == 0: # some boxes have a surface area of 0 like (0,76,100,76)
                continue
            possible_parents = []
            for box_, ii in zip(boxes, range(len(boxes))):
                if get_overlap_ratio(box, box_) == 1 and box != box_:
                    possible_parents.append(ii)
                    #print i, '-', ii
            I = boxes[i]
            put_here = []
            for pp in possible_parents:
                p_h = True
                level = nx.shortest_path_length(G,0,pp)+1
                if level in levels:
                    for window in levels[level]:
                        II = boxes[window]
                        if get_overlap_ratio(I, II) == 1:
                            p_h = False
                    if p_h == True:
                        put_here.append(pp)
                else:
                    put_here.append(pp)
            parent = min(put_here)
            level = nx.shortest_path_length(G,0,parent)+1
            if level in levels:
                if parent not in levels[level]:
                    levels[level].append(i)
                G.add_edge(i,parent)
            else:
                levels[level] = [i]
                G.add_edge(i,parent)

    return G, levels

    
def find_children(sucs, parent):
    if parent < len(sucs):
        if sucs.keys()[parent] == parent:
            children = sucs.values()[parent]
        else:
            for i in range(len(sucs)):
             if sucs.keys()[i] == parent:
               children = sucs.values()[i]
    else:
        for i in range(len(sucs)):
            if sucs.keys()[i] == parent:
               children = sucs.values()[i]
    
    return children
    
    
def find_root(pruned_boxes):
    all_coords = []
    G, levels = create_tree(pruned_boxes)
    #prune tree to only have levels which fully cover the image, tested
    nr_levels_covered = 100
    total_size = surface_area(pruned_boxes, levels[0])
    for level in levels:
        sa = surface_area(pruned_boxes, levels[level])
        sa_co = sa/total_size
        if sa_co != 1.0:
            G.remove_nodes_from(levels[level])
        else:
            nr_levels_covered = level
    levels = {k: levels[k] for k in range(0,nr_levels_covered + 1)}

    for patch in G.nodes():
        if pruned_boxes[patch] == []:
            print 'pruned_boxes[patch] == [] ', patch, pruned_boxes[patch], len(pruned_boxes)
        all_coords.append(pruned_boxes[patch])
    
    for level in levels:
        intersection_coords = extract_coords(levels[level], pruned_boxes)
        all_coords.extend(intersection_coords)
    a = np.array(all_coords)
    unique_coords = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
    to_delete = []
    for t_d,coord in enumerate(unique_coords):
        if coord[1] == coord[3] or coord[0]==coord[2]:
            to_delete.append(t_d)
            
    for t_d in reversed(to_delete):
        unique_coords = np.delete(unique_coords,t_d,0)
        
    new_index = unique_coords.tolist().index(pruned_boxes[0])
    return new_index
    
def sort_boxes(boxes, X):
    sorted_boxes = []
    sorted_features = []
    decorated = [((box[3]-box[1])*(box[2]-box[0]), i) for i, box in enumerate(boxes)]
    decorated.sort()
    for box, i in reversed(decorated):
        sorted_boxes.append(boxes[i])
        sorted_features.append(X[i])
    return sorted_boxes, sorted_features
    
def get_set_intersection(set_):
    intersection = get_intersection(set_[0], set_[1])
    if intersection == []:
       return []
    for s in set_[2:len(set_)]:
        intersection = get_intersection(intersection, s)
        if intersection == []:
            return []
    return intersection
      
def surface_area(boxes, boxes_level):
    if len(boxes_level) == 1:
        I = boxes[boxes_level[0]]
        return (I[3]-I[1])*(I[2]-I[0])
    surface_area = 0
    level_boxes = []
    index = {}
    nbrs = {}
    for i in boxes_level:
        level_boxes.append(boxes[i])
        
    combinations = list(itertools.combinations(boxes_level, 2)) 
    G = nx.Graph()
    
    G.add_edges_from(combinations)
    
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c])
        I = get_set_intersection(set_)
        if I == []:
            G.remove_edges_from([comb])
    
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
        I = [0,0,1000,1000]
        for c in base:
            I = get_intersection(boxes[c], I)
        if len(base)%2==1:
                surface_area += (I[3]-I[1])*(I[2]-I[0])
        elif len(base)%2==0:
            surface_area -= (I[3]-I[1])*(I[2]-I[0])
                
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None)))) 
            
    return surface_area
    
def surface_area_old(boxes, boxes_level):
    if len(boxes_level) == 1:
        I = boxes[boxes_level[0]]
        return (I[3]-I[1])*(I[2]-I[0])
    surface_area = 0
    level_boxes = []
    index = {}
    nbrs = {}
    for i in boxes_level:
        level_boxes.append(boxes[i])
        
    combinations = list(itertools.combinations(boxes_level, 2)) 
    G = nx.Graph()
    
    G.add_edges_from(combinations)
    
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c])
        I = get_set_intersection(set_)
        if I == []:
            G.remove_edges_from([comb])
    
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
        I = [0,0,1000,1000]
        for c in base:
            I = get_intersection(boxes[c], I)
        if len(base)%2==1:
                surface_area += (I[3]-I[1])*(I[2]-I[0])
        elif len(base)%2==0:
            surface_area -= (I[3]-I[1])*(I[2]-I[0])
                
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None)))) 
            
    return surface_area
    
def extract_coords(level_numbers, boxes):
    coords = []
    level_boxes = []
    for i in level_numbers:
        level_boxes.append(boxes[i])
        
    # create graph G from combinations possible        
    combinations = list(itertools.combinations(level_numbers, 2)) 
    G = nx.Graph()
    G.add_edges_from(combinations)
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c])
        I = get_set_intersection(set_)
        if I == []:
            G.remove_edges_from([comb])
    
    real_b = [b for b in boxes]
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
        if len(base) > length:
            length = len(base)
        I = [0,0,1100,1100]
        for c in base:
            if I != []:
               I = get_intersection(boxes[c], I)
        if I != []:
            #print I, I in np.array(real_b)
            #if I not in np.array(real_b):
            #    coords.append(I)
            coords.append(I)
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None))))
    
    return coords
    
def extract_coords_missing(level_numbers, boxes):
    coords = []
    level_boxes = []
    for i in level_numbers:
        level_boxes.append(boxes[i])
        
    # create graph G from combinations possible        
    combinations = list(itertools.combinations(level_numbers, 2)) 
    G = nx.Graph()
    G.add_edges_from(combinations)
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c])
        I = get_set_intersection(set_)
        if I == []:
            G.remove_edges_from([comb])
    
    real_b = [b for b in boxes]
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
        if len(base) > length:
            length = len(base)
        I = [0,0,1000,1000]
        for c in base:
            if I != []:
               I = get_intersection(boxes[c], I)
        if I != []:
            if I in real_b and len(base)>1:
                coords.append(I)
                    
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None))))
    
    return coords
