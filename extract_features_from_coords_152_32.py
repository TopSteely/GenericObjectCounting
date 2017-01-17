import os
import sys
sys.setrecursionlimit(50000)
import gzip
import pickle
import numpy as np
from scipy.misc import imread, imresize
from load import get_seperation, get_traineval_seperation, get_labels, get_features
#from caffe.proto import caffe_pb2
from get_intersection import get_intersection

import networkx as nx
import get_overlap_ratio
import itertools

from collections import deque
from itertools import chain, islice
import time

import theano
from theano import tensor as T

from sklearn import linear_model

import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ElemwiseSumLayer, DenseLayer, batch_norm, TransposedConv2DLayer, BatchNormLayer

from sklearn.preprocessing import MinMaxScaler

def get_set_intersection(set_):
    intersection = get_intersection(set_[0], set_[1])
    if intersection == []:
       return []
    for s in set_[2:len(set_)]:
        intersection = get_intersection(intersection, s)
        if intersection == []:
            return []
    return intersection
    
    
def extract_coords(level_numbers, boxes):
    coords = []
    level_boxes = []
    for i in level_numbers:
        level_boxes.append(boxes[i][0])
        
    # create graph G from combinations possible        
    combinations = list(itertools.combinations(level_numbers, 2)) 
    G = nx.Graph()
    G.add_edges_from(combinations)
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c][0])
        I = get_set_intersection(set_)
        if I == []:
            G.remove_edges_from([comb])
    
    real_b = [b[0] for b in boxes]
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
               I = get_intersection(boxes[c][0], I)
        if I != []:
            coords.append(I)
                    
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append((chain(base, [u]),
                          filter(nbrs[u].__contains__,
                                 islice(cnbrs, i + 1, None))))
    
    return coords
    
def create_tree(boxes):
    G = nx.Graph()
    levels = {}
    levels[0] = [0]
    G.add_node(0)
    if len(boxes) != 1:
        for box, i in zip(boxes[1:len(boxes)], range(1,len(boxes))):
            if (box[0][2]-box[0][0]) * (box[0][3]-box[0][1]) == 0: # some boxes have a surface area of 0 like (0,76,100,76)
                print box
                print 'surface area of box == 0', i
                continue
            possible_parents = []
            for box_, ii in zip(boxes, range(len(boxes))):
                if get_overlap_ratio.get_overlap_ratio(box[0], box_[0]) == 1 and box != box_:
                    possible_parents.append(ii)
                    #print i, '-', ii
            I = boxes[i][0]
            put_here = []
            for pp in possible_parents:
                p_h = True
                level = nx.shortest_path_length(G,0,pp)+1
                if level in levels:
                    for window in levels[level]:
                        II = boxes[window][0]
                        if get_overlap_ratio.get_overlap_ratio(I, II) == 1:
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
    
def sort_boxes(boxes, from_, to):
    sorted_boxes = []
    decorated = [((box[0][3]-box[0][1])*(box[0][2]-box[0][0]), i) for i, box in enumerate(boxes)]
    decorated.sort()
    for box, i in reversed(decorated):
        sorted_boxes.append(boxes[i])
    return sorted_boxes[from_:to]

def surface_area(boxes, boxes_level):
    if len(boxes_level) == 1:
        I = boxes[boxes_level[0]][0]
        return (I[3]-I[1])*(I[2]-I[0])
    surface_area = 0
    level_boxes = []
    index = {}
    nbrs = {}
    for i in boxes_level:
        level_boxes.append(boxes[i][0])
        
    combinations = list(itertools.combinations(boxes_level, 2)) 
    G = nx.Graph()
    
    G.add_edges_from(combinations)
    
    for comb in combinations:
        set_ = []
        for c in comb:
            set_.append(boxes[c][0])
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
            I = get_intersection(boxes[c][0], I)
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

flips = True
ignors = False
#import caffe
#from caffe.io import blobproto_to_array
#from caffe.proto.caffe_pb2 import BlobProto
#blob = BlobProto()
#data = open( 'ResNet_mean.binaryproto' , 'rb' ).read()
#blob.ParseFromString(data)
#mean = np.array( blobproto_to_array(blob) )
normalize = True
minibatch_size = 8

    
#if False:
#    #convert from caffe to Lasagne    
#    net_caffe = caffe.Net('ResNet-152-deploy.prototxt','ResNet-152-model.caffemodel',caffe.TEST)
#      
#    resnet = {}
#    resnet['input'] = InputLayer((1,3,224,224)) #minibatch size of 8
#    #resnet['lcn'] = LecunLCN.LecunLCN(X = T.tensor4('X'), image_shape=(1,3,224,224))
#    resnet['conv1'] = ConvLayer(resnet['input'], num_filters=64, filter_size=7, pad=3, stride=2, flip_filters=flips)
#    resnet['bn_conv1'] = BatchNormLayer(resnet['conv1'])
#    resnet['pool1'] = PoolLayer(resnet['bn_conv1'], pool_size=3, stride=2, mode='max', ignore_border=ignors)
#    resnet['res2a_branch1'] = ConvLayer(resnet['pool1'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn2a_branch1'] = BatchNormLayer(resnet['res2a_branch1'])
#    resnet['res2a_branch2a'] = ConvLayer(resnet['pool1'], num_filters=64, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn2a_branch2a'] = BatchNormLayer(resnet['res2a_branch2a'])
#    resnet['res2a_branch2b'] = ConvLayer(resnet['bn2a_branch2a'], num_filters=64, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn2a_branch2b'] = BatchNormLayer(resnet['res2a_branch2b'])
#    resnet['res2a_branch2c'] = ConvLayer(resnet['bn2a_branch2b'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn2a_branch2c'] = BatchNormLayer(resnet['res2a_branch2c'])
#    resnet['res2a'] = ElemwiseSumLayer((resnet['bn2a_branch2c'],resnet['bn2a_branch1']))
#    resnet['res2b_branch2a'] = ConvLayer(resnet['res2a'], num_filters=64, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn2b_branch2a'] = BatchNormLayer(resnet['res2b_branch2a'])
#    resnet['res2b_branch2b'] = ConvLayer(resnet['bn2b_branch2a'], num_filters=64, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn2b_branch2b'] = BatchNormLayer(resnet['res2b_branch2b'])
#    resnet['res2b_branch2c'] = ConvLayer(resnet['bn2b_branch2b'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn2b_branch2c'] = BatchNormLayer(resnet['res2b_branch2c'])
#    resnet['res2b'] = ElemwiseSumLayer((resnet['res2a'],resnet['bn2b_branch2c']))
#    resnet['res2c_branch2a'] = ConvLayer(resnet['res2b'], num_filters=64, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn2c_branch2a'] = BatchNormLayer(resnet['res2c_branch2a'])
#    resnet['res2c_branch2b'] = ConvLayer(resnet['bn2c_branch2a'], num_filters=64, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn2c_branch2b'] = BatchNormLayer(resnet['res2c_branch2b'])
#    resnet['res2c_branch2c'] = ConvLayer(resnet['bn2c_branch2b'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn2c_branch2c'] = BatchNormLayer(resnet['res2c_branch2c'])
#    resnet['res2c'] = ElemwiseSumLayer((resnet['res2b'],resnet['bn2c_branch2c']))
#    resnet['res3a_branch1'] = ConvLayer(resnet['res2c'], num_filters=512, filter_size=1, pad=0, stride=2, flip_filters=flips)
#    resnet['bn3a_branch1'] = BatchNormLayer(resnet['res3a_branch1'])
#    resnet['res3a_branch2a'] = ConvLayer(resnet['res2c'], num_filters=128, filter_size=1, pad=0, stride=2, flip_filters=flips)
#    resnet['bn3a_branch2a'] = BatchNormLayer(resnet['res3a_branch2a'])
#    resnet['res3a_branch2b'] = ConvLayer(resnet['bn3a_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3a_branch2b'] = BatchNormLayer(resnet['res3a_branch2b'])
#    resnet['res3a_branch2c'] = ConvLayer(resnet['bn3a_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3a_branch2c'] = BatchNormLayer(resnet['res3a_branch2c'])
#    resnet['res3a'] = ElemwiseSumLayer((resnet['bn3a_branch1'],resnet['bn3a_branch2c']))
#    resnet['res3b1_branch2a'] = ConvLayer(resnet['res3a'], num_filters=128, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b1_branch2a'] = BatchNormLayer(resnet['res3b1_branch2a'])    
#    resnet['res3b1_branch2b'] = ConvLayer(resnet['bn3b1_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3b1_branch2b'] = BatchNormLayer(resnet['res3b1_branch2b'])
#    resnet['res3b1_branch2c'] = ConvLayer(resnet['bn3b1_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b1_branch2c'] = BatchNormLayer(resnet['res3b1_branch2c'])
#    resnet['res3b1'] = ElemwiseSumLayer((resnet['res3a'],resnet['bn3b1_branch2c']))
#    resnet['res3b2_branch2a'] = ConvLayer(resnet['res3b1'], num_filters=128, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b2_branch2a'] = BatchNormLayer(resnet['res3b2_branch2a'])
#    resnet['res3b2_branch2b'] = ConvLayer(resnet['bn3b2_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3b2_branch2b'] = BatchNormLayer(resnet['res3b2_branch2b'])
#    resnet['res3b2_branch2c'] = ConvLayer(resnet['bn3b2_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b2_branch2c'] = BatchNormLayer(resnet['res3b2_branch2c'])
#    resnet['res3b2'] = ElemwiseSumLayer((resnet['res3b1'],resnet['bn3b2_branch2c']))
#    resnet['res3b3_branch2a'] = ConvLayer(resnet['res3b2'], num_filters=128, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b3_branch2a'] = BatchNormLayer(resnet['res3b3_branch2a'])
#    resnet['res3b3_branch2b'] = ConvLayer(resnet['bn3b3_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3b3_branch2b'] = BatchNormLayer(resnet['res3b3_branch2b'])
#    resnet['res3b3_branch2c'] = ConvLayer(resnet['bn3b3_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b3_branch2c'] = BatchNormLayer(resnet['res3b3_branch2c'])
#    resnet['res3b3'] = ElemwiseSumLayer((resnet['res3b2'],resnet['bn3b3_branch2c']))
#    resnet['res3b4_branch2a'] = ConvLayer(resnet['res3b3'], num_filters=128, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b4_branch2a'] = BatchNormLayer(resnet['res3b4_branch2a'])
#    resnet['res3b4_branch2b'] = ConvLayer(resnet['bn3b4_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3b4_branch2b'] = BatchNormLayer(resnet['res3b4_branch2b'])
#    resnet['res3b4_branch2c'] = ConvLayer(resnet['bn3b4_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b4_branch2c'] = BatchNormLayer(resnet['res3b4_branch2c'])
#    resnet['res3b4'] = ElemwiseSumLayer((resnet['res3b3'],resnet['bn3b4_branch2c']))
#    resnet['res3b5_branch2a'] = ConvLayer(resnet['res3b4'], num_filters=128, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b5_branch2a'] = BatchNormLayer(resnet['res3b5_branch2a'])
#    resnet['res3b5_branch2b'] = ConvLayer(resnet['bn3b5_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3b5_branch2b'] = BatchNormLayer(resnet['res3b5_branch2b'])
#    resnet['res3b5_branch2c'] = ConvLayer(resnet['bn3b5_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b5_branch2c'] = BatchNormLayer(resnet['res3b5_branch2c'])
#    resnet['res3b5'] = ElemwiseSumLayer((resnet['res3b4'],resnet['bn3b5_branch2c']))
#    resnet['res3b6_branch2a'] = ConvLayer(resnet['res3b5'], num_filters=128, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b6_branch2a'] = BatchNormLayer(resnet['res3b6_branch2a'])
#    resnet['res3b6_branch2b'] = ConvLayer(resnet['bn3b6_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3b6_branch2b'] = BatchNormLayer(resnet['res3b6_branch2b'])
#    resnet['res3b6_branch2c'] = ConvLayer(resnet['bn3b6_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b6_branch2c'] = BatchNormLayer(resnet['res3b6_branch2c'])
#    resnet['res3b6'] = ElemwiseSumLayer((resnet['res3b5'],resnet['bn3b6_branch2c']))
#    resnet['res3b7_branch2a'] = ConvLayer(resnet['res3b6'], num_filters=128, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b7_branch2a'] = BatchNormLayer(resnet['res3b5_branch2a'])
#    resnet['res3b7_branch2b'] = ConvLayer(resnet['bn3b5_branch2a'], num_filters=128, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn3b7_branch2b'] = BatchNormLayer(resnet['res3b5_branch2b'])
#    resnet['res3b7_branch2c'] = ConvLayer(resnet['bn3b5_branch2b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn3b7_branch2c'] = BatchNormLayer(resnet['res3b5_branch2c'])
#    resnet['res3b7'] = ElemwiseSumLayer((resnet['res3b6'],resnet['bn3b7_branch2c']))
#    
#    
#    resnet['res4a_branch1'] = ConvLayer(resnet['res3b3'], num_filters=1024, filter_size=1, pad=0, stride =2, flip_filters=flips)
#    resnet['bn4a_branch1'] = BatchNormLayer(resnet['res4a_branch1'])
#    resnet['res4a_branch2a'] = ConvLayer(resnet['res3b3'], num_filters=256, filter_size=1, pad=0, stride=2, flip_filters=flips)
#    resnet['bn4a_branch2a'] = BatchNormLayer(resnet['res4a_branch2a'])
#    resnet['res4a_branch2b'] = ConvLayer(resnet['bn4a_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4a_branch2b'] = BatchNormLayer(resnet['res4a_branch2b'])
#    resnet['res4a_branch2c'] = ConvLayer(resnet['bn4a_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4a_branch2c'] = BatchNormLayer(resnet['res4a_branch2c'])
#    resnet['res4a'] = ElemwiseSumLayer((resnet['bn4a_branch1'],resnet['bn4a_branch2c']))
#    resnet['res4b1_branch2a'] = ConvLayer(resnet['res4a'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b1_branch2a'] = BatchNormLayer(resnet['res4b1_branch2a'])
#    resnet['res4b1_branch2b'] = ConvLayer(resnet['bn4b1_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b1_branch2b'] = BatchNormLayer(resnet['res4b1_branch2b'])
#    resnet['res4b1_branch2c'] = ConvLayer(resnet['bn4b1_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b1_branch2c'] = BatchNormLayer(resnet['res4b1_branch2c'])
#    resnet['res4b1'] = ElemwiseSumLayer((resnet['bn4b1_branch2c'],resnet['res4a']))
#    resnet['res4b2_branch2a'] = ConvLayer(resnet['res4b1'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b2_branch2a'] = BatchNormLayer(resnet['res4b2_branch2a'])
#    resnet['res4b2_branch2b'] = ConvLayer(resnet['bn4b2_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b2_branch2b'] = BatchNormLayer(resnet['res4b2_branch2b'])
#    resnet['res4b2_branch2c'] = ConvLayer(resnet['bn4b2_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b2_branch2c'] = BatchNormLayer(resnet['res4b2_branch2c'])
#    resnet['res4b2'] = ElemwiseSumLayer((resnet['bn4b2_branch2c'],resnet['res4b1']))
#    resnet['res4b3_branch2a'] = ConvLayer(resnet['res4b2'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b3_branch2a'] = BatchNormLayer(resnet['res4b3_branch2a'])
#    resnet['res4b3_branch2b'] = ConvLayer(resnet['bn4b3_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b3_branch2b'] = BatchNormLayer(resnet['res4b3_branch2b'])
#    resnet['res4b3_branch2c'] = ConvLayer(resnet['bn4b3_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b3_branch2c'] = BatchNormLayer(resnet['res4b3_branch2c'])
#    resnet['res4b3'] = ElemwiseSumLayer((resnet['bn4b3_branch2c'],resnet['res4b2']))
#    resnet['res4b4_branch2a'] = ConvLayer(resnet['res4b3'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b4_branch2a'] = BatchNormLayer(resnet['res4b4_branch2a'])
#    resnet['res4b4_branch2b'] = ConvLayer(resnet['bn4b4_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b4_branch2b'] = BatchNormLayer(resnet['res4b4_branch2b'])
#    resnet['res4b4_branch2c'] = ConvLayer(resnet['bn4b4_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b4_branch2c'] = BatchNormLayer(resnet['res4b4_branch2c'])
#    resnet['res4b4'] = ElemwiseSumLayer((resnet['bn4b4_branch2c'],resnet['res4b3']))
#    resnet['res4b5_branch2a'] = ConvLayer(resnet['res4b4'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b5_branch2a'] = BatchNormLayer(resnet['res4b5_branch2a'])
#    resnet['res4b5_branch2b'] = ConvLayer(resnet['bn4b5_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b5_branch2b'] = BatchNormLayer(resnet['res4b5_branch2b'])
#    resnet['res4b5_branch2c'] = ConvLayer(resnet['bn4b5_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b5_branch2c'] = BatchNormLayer(resnet['res4b5_branch2c'])
#    resnet['res4b5'] = ElemwiseSumLayer((resnet['bn4b5_branch2c'],resnet['res4b4']))
#    resnet['res4b6_branch2a'] = ConvLayer(resnet['res4b5'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b6_branch2a'] = BatchNormLayer(resnet['res4b6_branch2a'])
#    resnet['res4b6_branch2b'] = ConvLayer(resnet['bn4b6_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b6_branch2b'] = BatchNormLayer(resnet['res4b6_branch2b'])
#    resnet['res4b6_branch2c'] = ConvLayer(resnet['bn4b6_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b6_branch2c'] = BatchNormLayer(resnet['res4b6_branch2c'])
#    resnet['res4b6'] = ElemwiseSumLayer((resnet['bn4b6_branch2c'],resnet['res4b5']))
#    resnet['res4b7_branch2a'] = ConvLayer(resnet['res4b6'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b7_branch2a'] = BatchNormLayer(resnet['res4b7_branch2a'])
#    resnet['res4b7_branch2b'] = ConvLayer(resnet['bn4b7_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b7_branch2b'] = BatchNormLayer(resnet['res4b7_branch2b'])
#    resnet['res4b7_branch2c'] = ConvLayer(resnet['bn4b7_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b7_branch2c'] = BatchNormLayer(resnet['res4b7_branch2c'])
#    resnet['res4b7'] = ElemwiseSumLayer((resnet['res4b7_branch2c'],resnet['res4b6']))
#    resnet['res4b8_branch2a'] = ConvLayer(resnet['res4b7'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b8_branch2a'] = BatchNormLayer(resnet['res4b8_branch2a'])
#    resnet['res4b8_branch2b'] = ConvLayer(resnet['bn4b8_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b8_branch2b'] = BatchNormLayer(resnet['res4b8_branch2b'])
#    resnet['res4b8_branch2c'] = ConvLayer(resnet['bn4b8_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b8_branch2c'] = BatchNormLayer(resnet['res4b8_branch2c'])
#    resnet['res4b8'] = ElemwiseSumLayer((resnet['bn4b8_branch2c'],resnet['res4b7']))
#    resnet['res4b9_branch2a'] = ConvLayer(resnet['res4b8'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b9_branch2a'] = BatchNormLayer(resnet['res4b9_branch2a'])
#    resnet['res4b9_branch2b'] = ConvLayer(resnet['bn4b9_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b9_branch2b'] = BatchNormLayer(resnet['res4b9_branch2b'])
#    resnet['res4b9_branch2c'] = ConvLayer(resnet['bn4b9_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b9_branch2c'] = BatchNormLayer(resnet['res4b9_branch2c'])
#    resnet['res4b9'] = ElemwiseSumLayer((resnet['bn4b9_branch2c'],resnet['res4b8']))
#    resnet['res4b10_branch2a'] = ConvLayer(resnet['res4b9'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b10_branch2a'] = BatchNormLayer(resnet['res4b10_branch2a'])
#    resnet['res4b10_branch2b'] = ConvLayer(resnet['bn4b10_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b10_branch2b'] = BatchNormLayer(resnet['res4b10_branch2b'])
#    resnet['res4b10_branch2c'] = ConvLayer(resnet['bn4b10_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b10_branch2c'] = BatchNormLayer(resnet['res4b10_branch2c'])
#    resnet['res4b10'] = ElemwiseSumLayer((resnet['bn4b10_branch2c'],resnet['res4b9']))
#    resnet['res4b11_branch2a'] = ConvLayer(resnet['res4b10'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b11_branch2a'] = BatchNormLayer(resnet['res4b11_branch2a'])
#    resnet['res4b11_branch2b'] = ConvLayer(resnet['bn4b11_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b11_branch2b'] = BatchNormLayer(resnet['res4b11_branch2b'])
#    resnet['res4b11_branch2c'] = ConvLayer(resnet['bn4b11_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b11_branch2c'] = BatchNormLayer(resnet['res4b11_branch2c'])
#    resnet['res4b11'] = ElemwiseSumLayer((resnet['bn4b11_branch2c'],resnet['res4b10']))
#    resnet['res4b12_branch2a'] = ConvLayer(resnet['res4b11'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b12_branch2a'] = BatchNormLayer(resnet['res4b12_branch2a'])
#    resnet['res4b12_branch2b'] = ConvLayer(resnet['bn4b12_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b12_branch2b'] = BatchNormLayer(resnet['res4b12_branch2b'])
#    resnet['res4b12_branch2c'] = ConvLayer(resnet['bn4b12_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b12_branch2c'] = BatchNormLayer(resnet['res4b12_branch2c'])
#    resnet['res4b12'] = ElemwiseSumLayer((resnet['bn4b12_branch2c'],resnet['res4b11']))
#    resnet['res4b13_branch2a'] = ConvLayer(resnet['res4b12'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b13_branch2a'] = BatchNormLayer(resnet['res4b13_branch2a'])
#    resnet['res4b13_branch2b'] = ConvLayer(resnet['bn4b13_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b13_branch2b'] = BatchNormLayer(resnet['res4b13_branch2b'])
#    resnet['res4b13_branch2c'] = ConvLayer(resnet['bn4b13_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b13_branch2c'] = BatchNormLayer(resnet['res4b13_branch2c'])
#    resnet['res4b13'] = ElemwiseSumLayer((resnet['bn4b13_branch2c'],resnet['res4b12']))
#    resnet['res4b14_branch2a'] = ConvLayer(resnet['res4b13'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b14_branch2a'] = BatchNormLayer(resnet['res4b14_branch2a'])
#    resnet['res4b14_branch2b'] = ConvLayer(resnet['bn4b14_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b14_branch2b'] = BatchNormLayer(resnet['res4b14_branch2b'])
#    resnet['res4b14_branch2c'] = ConvLayer(resnet['bn4b14_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b14_branch2c'] = BatchNormLayer(resnet['res4b14_branch2c'])
#    resnet['res4b14'] = ElemwiseSumLayer((resnet['bn4b14_branch2c'],resnet['res4b13']))
#    resnet['res4b15_branch2a'] = ConvLayer(resnet['res4b14'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b15_branch2a'] = BatchNormLayer(resnet['res4b15_branch2a'])
#    resnet['res4b15_branch2b'] = ConvLayer(resnet['bn4b15_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b15_branch2b'] = BatchNormLayer(resnet['res4b15_branch2b'])
#    resnet['res4b15_branch2c'] = ConvLayer(resnet['bn4b15_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b15_branch2c'] = BatchNormLayer(resnet['res4b15_branch2c'])
#    resnet['res4b15'] = ElemwiseSumLayer((resnet['bn4b15_branch2c'],resnet['res4b14']))
#    resnet['res4b16_branch2a'] = ConvLayer(resnet['res4b15'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b16_branch2a'] = BatchNormLayer(resnet['res4b16_branch2a'])
#    resnet['res4b16_branch2b'] = ConvLayer(resnet['bn4b16_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b16_branch2b'] = BatchNormLayer(resnet['res4b16_branch2b'])
#    resnet['res4b16_branch2c'] = ConvLayer(resnet['bn4b16_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b16_branch2c'] = BatchNormLayer(resnet['res4b16_branch2c'])
#    resnet['res4b16'] = ElemwiseSumLayer((resnet['res4b16_branch2c'],resnet['res4b15']))
#    resnet['res4b17_branch2a'] = ConvLayer(resnet['res4b16'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b17_branch2a'] = BatchNormLayer(resnet['res4b17_branch2a'])
#    resnet['res4b17_branch2b'] = ConvLayer(resnet['bn4b17_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b17_branch2b'] = BatchNormLayer(resnet['res4b17_branch2b'])
#    resnet['res4b17_branch2c'] = ConvLayer(resnet['bn4b17_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b17_branch2c'] = BatchNormLayer(resnet['res4b17_branch2c'])
#    resnet['res4b17'] = ElemwiseSumLayer((resnet['bn4b17_branch2c'],resnet['res4b16']))
#    resnet['res4b18_branch2a'] = ConvLayer(resnet['res4b17'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b18_branch2a'] = BatchNormLayer(resnet['res4b18_branch2a'])
#    resnet['res4b18_branch2b'] = ConvLayer(resnet['bn4b18_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b18_branch2b'] = BatchNormLayer(resnet['res4b18_branch2b'])
#    resnet['res4b18_branch2c'] = ConvLayer(resnet['bn4b18_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b18_branch2c'] = BatchNormLayer(resnet['res4b18_branch2c'])
#    resnet['res4b18'] = ElemwiseSumLayer((resnet['bn4b18_branch2c'],resnet['res4b17']))
#    resnet['res4b19_branch2a'] = ConvLayer(resnet['res4b18'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b19_branch2a'] = BatchNormLayer(resnet['res4b19_branch2a'])
#    resnet['res4b19_branch2b'] = ConvLayer(resnet['bn4b19_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b19_branch2b'] = BatchNormLayer(resnet['res4b19_branch2b'])
#    resnet['res4b19_branch2c'] = ConvLayer(resnet['bn4b19_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b19_branch2c'] = BatchNormLayer(resnet['res4b19_branch2c'])
#    resnet['res4b19'] = ElemwiseSumLayer((resnet['bn4b19_branch2c'],resnet['res4b18']))
#    resnet['res4b20_branch2a'] = ConvLayer(resnet['res4b19'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b20_branch2a'] = BatchNormLayer(resnet['res4b20_branch2a'])
#    resnet['res4b20_branch2b'] = ConvLayer(resnet['bn4b20_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b20_branch2b'] = BatchNormLayer(resnet['res4b20_branch2b'])
#    resnet['res4b20_branch2c'] = ConvLayer(resnet['bn4b20_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b20_branch2c'] = BatchNormLayer(resnet['res4b20_branch2c'])
#    resnet['res4b20'] = ElemwiseSumLayer((resnet['bn4b20_branch2c'],resnet['res4b19']))
#    resnet['res4b21_branch2a'] = ConvLayer(resnet['res4b20'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b21_branch2a'] = BatchNormLayer(resnet['res4b21_branch2a'])
#    resnet['res4b21_branch2b'] = ConvLayer(resnet['bn4b21_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b21_branch2b'] = BatchNormLayer(resnet['res4b21_branch2b'])
#    resnet['res4b21_branch2c'] = ConvLayer(resnet['bn4b21_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b21_branch2c'] = BatchNormLayer(resnet['res4b21_branch2c'])
#    resnet['res4b21'] = ElemwiseSumLayer((resnet['bn4b21_branch2c'],resnet['res4b20']))
#    resnet['res4b22_branch2a'] = ConvLayer(resnet['res4b21'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b22_branch2a'] = BatchNormLayer(resnet['res4b22_branch2a'])
#    resnet['res4b22_branch2b'] = ConvLayer(resnet['bn4b22_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b22_branch2b'] = BatchNormLayer(resnet['res4b22_branch2b'])
#    resnet['res4b22_branch2c'] = ConvLayer(resnet['bn4b22_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b22_branch2c'] = BatchNormLayer(resnet['res4b22_branch2c'])
#    resnet['res4b22'] = ElemwiseSumLayer((resnet['bn4b22_branch2c'],resnet['res4b21']))
#    #new to resnet152
#    resnet['res4b23_branch2a'] = ConvLayer(resnet['res4b22'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b23_branch2a'] = BatchNormLayer(resnet['res4b23_branch2a'])
#    resnet['res4b23_branch2b'] = ConvLayer(resnet['bn4b23_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b23_branch2b'] = BatchNormLayer(resnet['res4b23_branch2b'])
#    resnet['res4b23_branch2c'] = ConvLayer(resnet['bn4b23_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b23_branch2c'] = BatchNormLayer(resnet['res4b23_branch2c'])
#    resnet['res4b23'] = ElemwiseSumLayer((resnet['bn4b23_branch2c'],resnet['res4b22']))
#    resnet['res4b24_branch2a'] = ConvLayer(resnet['res4b23'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b24_branch2a'] = BatchNormLayer(resnet['res4b24_branch2a'])
#    resnet['res4b24_branch2b'] = ConvLayer(resnet['bn4b24_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b24_branch2b'] = BatchNormLayer(resnet['res4b24_branch2b'])
#    resnet['res4b24_branch2c'] = ConvLayer(resnet['bn4b24_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b24_branch2c'] = BatchNormLayer(resnet['res4b24_branch2c'])
#    resnet['res4b24'] = ElemwiseSumLayer((resnet['bn4b24_branch2c'],resnet['res4b23']))
#    resnet['res4b25_branch2a'] = ConvLayer(resnet['res4b24'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b25_branch2a'] = BatchNormLayer(resnet['res4b25_branch2a'])
#    resnet['res4b25_branch2b'] = ConvLayer(resnet['bn4b25_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b25_branch2b'] = BatchNormLayer(resnet['res4b25_branch2b'])
#    resnet['res4b25_branch2c'] = ConvLayer(resnet['bn4b25_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b25_branch2c'] = BatchNormLayer(resnet['res4b25_branch2c'])
#    resnet['res4b25'] = ElemwiseSumLayer((resnet['bn4b25_branch2c'],resnet['res4b24']))
#    resnet['res4b26_branch2a'] = ConvLayer(resnet['res4b25'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b26_branch2a'] = BatchNormLayer(resnet['res4b26_branch2a'])
#    resnet['res4b26_branch2b'] = ConvLayer(resnet['bn4b26_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b26_branch2b'] = BatchNormLayer(resnet['res4b26_branch2b'])
#    resnet['res4b26_branch2c'] = ConvLayer(resnet['bn4b26_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b26_branch2c'] = BatchNormLayer(resnet['res4b26_branch2c'])
#    resnet['res4b26'] = ElemwiseSumLayer((resnet['bn4b26_branch2c'],resnet['res4b25']))
#    resnet['res4b27_branch2a'] = ConvLayer(resnet['res4b26'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b27_branch2a'] = BatchNormLayer(resnet['res4b27_branch2a'])
#    resnet['res4b27_branch2b'] = ConvLayer(resnet['bn4b27_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b27_branch2b'] = BatchNormLayer(resnet['res4b27_branch2b'])
#    resnet['res4b27_branch2c'] = ConvLayer(resnet['bn4b27_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b27_branch2c'] = BatchNormLayer(resnet['res4b27_branch2c'])
#    resnet['res4b27'] = ElemwiseSumLayer((resnet['bn4b22_branch2c'],resnet['res4b26']))
#    resnet['res4b28_branch2a'] = ConvLayer(resnet['res4b27'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b28_branch2a'] = BatchNormLayer(resnet['res4b28_branch2a'])
#    resnet['res4b28_branch2b'] = ConvLayer(resnet['bn4b28_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b28_branch2b'] = BatchNormLayer(resnet['res4b28_branch2b'])
#    resnet['res4b28_branch2c'] = ConvLayer(resnet['bn4b28_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b28_branch2c'] = BatchNormLayer(resnet['res4b28_branch2c'])
#    resnet['res4b28'] = ElemwiseSumLayer((resnet['bn4b28_branch2c'],resnet['res4b27']))
#    resnet['res4b29_branch2a'] = ConvLayer(resnet['res4b28'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b29_branch2a'] = BatchNormLayer(resnet['res4b29_branch2a'])
#    resnet['res4b29_branch2b'] = ConvLayer(resnet['bn4b29_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b29_branch2b'] = BatchNormLayer(resnet['res4b29_branch2b'])
#    resnet['res4b29_branch2c'] = ConvLayer(resnet['bn4b29_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b29_branch2c'] = BatchNormLayer(resnet['res4b29_branch2c'])
#    resnet['res4b29'] = ElemwiseSumLayer((resnet['bn4b29_branch2c'],resnet['res4b28']))
#    resnet['res4b30_branch2a'] = ConvLayer(resnet['res4b29'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b30_branch2a'] = BatchNormLayer(resnet['res4b30_branch2a'])
#    resnet['res4b30_branch2b'] = ConvLayer(resnet['bn4b30_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b30_branch2b'] = BatchNormLayer(resnet['res4b30_branch2b'])
#    resnet['res4b30_branch2c'] = ConvLayer(resnet['bn4b30_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b30_branch2c'] = BatchNormLayer(resnet['res4b30_branch2c'])
#    resnet['res4b30'] = ElemwiseSumLayer((resnet['bn4b30_branch2c'],resnet['res4b29']))
#    resnet['res4b31_branch2a'] = ConvLayer(resnet['res4b30'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b31_branch2a'] = BatchNormLayer(resnet['res4b31_branch2a'])
#    resnet['res4b31_branch2b'] = ConvLayer(resnet['bn4b31_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b31_branch2b'] = BatchNormLayer(resnet['res4b31_branch2b'])
#    resnet['res4b31_branch2c'] = ConvLayer(resnet['bn4b31_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b31_branch2c'] = BatchNormLayer(resnet['res4b31_branch2c'])
#    resnet['res4b31'] = ElemwiseSumLayer((resnet['bn4b31_branch2c'],resnet['res4b30']))
#    resnet['res4b32_branch2a'] = ConvLayer(resnet['res4b31'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b32_branch2a'] = BatchNormLayer(resnet['res4b32_branch2a'])
#    resnet['res4b32_branch2b'] = ConvLayer(resnet['bn4b32_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b32_branch2b'] = BatchNormLayer(resnet['res4b32_branch2b'])
#    resnet['res4b32_branch2c'] = ConvLayer(resnet['bn4b32_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b32_branch2c'] = BatchNormLayer(resnet['res4b32_branch2c'])
#    resnet['res4b32'] = ElemwiseSumLayer((resnet['bn4b32_branch2c'],resnet['res4b31']))
#    resnet['res4b33_branch2a'] = ConvLayer(resnet['res4b32'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b33_branch2a'] = BatchNormLayer(resnet['res4b33_branch2a'])
#    resnet['res4b33_branch2b'] = ConvLayer(resnet['bn4b33_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b33_branch2b'] = BatchNormLayer(resnet['res4b33_branch2b'])
#    resnet['res4b33_branch2c'] = ConvLayer(resnet['bn4b33_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b33_branch2c'] = BatchNormLayer(resnet['res4b33_branch2c'])
#    resnet['res4b33'] = ElemwiseSumLayer((resnet['bn4b33_branch2c'],resnet['res4b32']))
#    resnet['res4b34_branch2a'] = ConvLayer(resnet['res4b33'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b34_branch2a'] = BatchNormLayer(resnet['res4b34_branch2a'])
#    resnet['res4b34_branch2b'] = ConvLayer(resnet['bn4b34_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b34_branch2b'] = BatchNormLayer(resnet['res4b34_branch2b'])
#    resnet['res4b34_branch2c'] = ConvLayer(resnet['bn4b34_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b34_branch2c'] = BatchNormLayer(resnet['res4b34_branch2c'])
#    resnet['res4b34'] = ElemwiseSumLayer((resnet['bn4b34_branch2c'],resnet['res4b33']))
#    resnet['res4b35_branch2a'] = ConvLayer(resnet['res4b34'], num_filters=256, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b35_branch2a'] = BatchNormLayer(resnet['res4b35_branch2a'])
#    resnet['res4b35_branch2b'] = ConvLayer(resnet['bn4b35_branch2a'], num_filters=256, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn4b35_branch2b'] = BatchNormLayer(resnet['res4b35_branch2b'])
#    resnet['res4b35_branch2c'] = ConvLayer(resnet['bn4b35_branch2b'], num_filters=1024, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn4b35_branch2c'] = BatchNormLayer(resnet['res4b35_branch2c'])
#    resnet['res4b35'] = ElemwiseSumLayer((resnet['bn4b35_branch2c'],resnet['res4b34']))
#    
#    resnet['res5a_branch1'] = ConvLayer(resnet['res4b35'], num_filters=2048, filter_size=1, pad=0, stride=2, flip_filters=flips)
#    resnet['bn5a_branch1'] = BatchNormLayer(resnet['res5a_branch1'])
#    resnet['res5a_branch2a'] = ConvLayer(resnet['res4b35'], num_filters=512, filter_size=1, pad=0, stride=2, flip_filters=flips)
#    resnet['bn5a_branch2a'] = BatchNormLayer(resnet['res5a_branch2a'])
#    resnet['res5a_branch2b'] = ConvLayer(resnet['bn5a_branch2a'], num_filters=512, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn5a_branch2b'] = BatchNormLayer(resnet['res5a_branch2b'])
#    resnet['res5a_branch2c'] = ConvLayer(resnet['bn5a_branch2b'], num_filters=2048, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn5a_branch2c'] = BatchNormLayer(resnet['res5a_branch2c'])
#    resnet['res5a'] = ElemwiseSumLayer((resnet['bn5a_branch2c'],resnet['bn5a_branch1']))
#    resnet['res5b_branch2a'] = ConvLayer(resnet['res5a'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn5b_branch2a'] = BatchNormLayer(resnet['res5b_branch2a'])
#    resnet['res5b_branch2b'] = ConvLayer(resnet['bn5b_branch2a'], num_filters=512, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn5b_branch2b'] = BatchNormLayer(resnet['res5b_branch2b'])
#    resnet['res5b_branch2c'] = ConvLayer(resnet['bn5b_branch2b'], num_filters=2048, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn5b_branch2c'] = BatchNormLayer(resnet['res5b_branch2c'])
#    resnet['res5b'] = ElemwiseSumLayer((resnet['bn5b_branch2c'],resnet['res5a']))
#    resnet['res5c_branch2a'] = ConvLayer(resnet['res5b'], num_filters=512, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn5c_branch2a'] = BatchNormLayer(resnet['res5c_branch2a'])
#    resnet['res5c_branch2b'] = ConvLayer(resnet['bn5c_branch2a'], num_filters=512, filter_size=3, pad=1, flip_filters=flips)
#    resnet['bn5c_branch2b'] = BatchNormLayer(resnet['res5c_branch2b'])
#    resnet['res5c_branch2c'] = ConvLayer(resnet['bn5c_branch2b'], num_filters=2048, filter_size=1, pad=0, flip_filters=flips)
#    resnet['bn5c_branch2c'] = BatchNormLayer(resnet['res5c_branch2c'])
#    resnet['res5c'] = ElemwiseSumLayer((resnet['bn5c_branch2c'],resnet['res5b']))
#    resnet['pool5'] = PoolLayer(resnet['res5c'], pool_size=7, stride=1, mode='average_exc_pad', ignore_border=ignors)
#    
#    #print inspect.getargspec(DenseLayer)
#    
#    resnet['fc1000'] = DenseLayer(resnet['pool5'], num_units=1000)
#    
#        
#    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))
#    
#    #copy parameters
#    for name, layer in resnet.items():
#        try:
#            if name.startswith('bn'):
#                tmp = name.split('bn')[1]
#                layer.beta.set_value(layers_caffe['scale'+tmp].blobs[1].data)
#                layer.gamma.set_value(layers_caffe['scale'+tmp].blobs[0].data)
#                layer.mean.set_value(layers_caffe[name].blobs[0].data)
#                layer.inv_std.set_value((1./np.sqrt(layers_caffe[name].blobs[1].data  + 1e-5)).astype(theano.config.floatX))
#            elif name == 'lcn':
#                continue
#            elif name == 'fc1000':
#                layer.W.set_value(layers_caffe['fc1000'].blobs[0].data.T)
#                layer.b.set_value(layers_caffe['fc1000'].blobs[1].data.T)
##            if name == 'bn2a_branch2b':
##                print len(layers_caffe['bn2a_branch2b'].blobs), len(layers_caffe['bn2a_branch2b'].blobs[0])
#            elif len(layers_caffe[name].blobs)== 1:
#                print name, '1'
#                #layer.W.set_value(layers_caffe[name].blobs[0].data[:,:,::-1,::-1])
#                layer.W.set_value(layers_caffe[name].blobs[0].data)
#            elif len(layers_caffe[name].blobs) == 2:
#                print name, '2'
#            #    layer.W.set_value(layers_caffe[name].blobs[0].data)
#            #    layer.b.set_value(layers_caffe[name].blobs[1].data)       
#            else:
#                print name, len(layers_caffe[name].blobs)
#        except AttributeError:
#            continue
#        
#    pickle.dump( resnet, open( "resnet_152.p", "wb" ) )
#else:
    
print 'starting'
file = open("resnet_152.p","rb")
resnet = pickle.load(file)
print 'loaded resnet152'
file = open("resnet_152_32.p","rb")
resnet32 = pickle.load(file)
print 'loaded resnet152_32'
X = T.tensor4('X')
Y = T.ivector('y')
data = 'pascal'
#
output_caffe = lasagne.layers.get_output(resnet['fc1000'], X, deterministic=True)
features_caffe = theano.function(inputs=[X], outputs=output_caffe)
output_caffe32 = lasagne.layers.get_output(resnet32['fc1000'], X, deterministic=True)
features_caffe32 = theano.function(inputs=[X], outputs=output_caffe32)

rrange = 600000 if data == 'mscoco' else 9963

if data == 'trancos':
    a = 3
else:
    a = 1

for a_i in range(1,a+1):
    start = time.time()
    for img_nr in range(9787,9964):
        index_feat = 1
        print img_nr
        coords_to_extract = []
        batch_feature = []
        feat_full_all = []
        feat_last_conv_all = []
        #load image
        if data == 'trancos':
            if os.path.isfile('/var/node436/local/tstahl/TRANCOS_v3/images/image-'+str(a_i) +'-' + (format(img_nr, "06d")) + '.jpg'):
                img = imread('/var/node436/local/tstahl/TRANCOS_v3/images/image-'+str(a_i) +'-' + (format(img_nr, "06d")) + '.jpg')
            else:
                print 'warning: /var/node436/local/tstahl/mscoco/train2014/images/image-'+str(a_i) +'-' + (format(img_nr, "06d")) + '.jpg'
                continue
        elif data == 'mscoco':
            if os.path.isfile('/var/node436/local/tstahl/mscoco/train2014/COCO_train2014_' + (format(img_nr, "012d")) + '.jpg'):
                img = imread('/var/node436/local/tstahl/mscoco/train2014/COCO_train2014_' + (format(img_nr, "012d")) + '.jpg')
            else:
                print 'warning: /var/node436/local/tstahl/mscoco/train2014/COCO_train2014_' + (format(img_nr, "012d")) + '.jpg'
                continue
        elif data == 'pascal':
            if os.path.isfile('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg'):
                img = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
            else:
                print 'warning: /var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg doesnt exist'
        if data == 'trancos':
            if os.path.isfile('/var/node436/local/tstahl/TRANCOS_v3/TRANCOS/SS_Boxes/'+str(a_i) +'-' + (format(img_nr, "06d")) +'.txt'):
                    f = open('/var/node436/local/tstahl/TRANCOS_v3/TRANCOS/SS_Boxes/'+str(a_i) +'-' + (format(img_nr, "06d")) +'.txt', 'r')
            else:
                print 'warning /var/node436/local/tstahl/TRANCOS_v3/TRANCOS/SS_Boxes/'+str(a_i) +'-' + (format(img_nr, "06d")) +'.txt doesnt exist'
        elif data == 'pascal':
            if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt'):
                    f = open('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt', 'r')
            else:
                print 'warning /var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt doesnt exist'
        else:
            if os.path.isfile('/var/node436/local/tstahl/mscoco/SS_Boxes/'+ (format(img_nr, "012d")) +'.txt'):
                    f = open('/var/node436/local/tstahl/mscoco/SS_Boxes/'+ (format(img_nr, "012d")) +'.txt', 'r')
            else:
                print 'warning /var/node436/local/tstahl/mscoco/SS_Boxes/'+ (format(img_nr, "012d")) +'.txt doesnt exist'
                continue
            
        boxes = []
        for i_n, line in enumerate(f):
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append([coord])
        f.close()
        boxes = sort_boxes(boxes, 0,5000)
            
            
        # after extraction of coords, create tree and extract needed intersections
        # create_tree
        G, levels = create_tree(boxes)
        
        #prune tree to only have levels which fully cover the image, tested
        nr_levels_covered = 100
        total_size = surface_area(boxes, levels[0])
        for level in levels:
            sa = surface_area(boxes, levels[level])
            sa_co = sa/total_size
            if sa_co != 1.0:
                G.remove_nodes_from(levels[level])
            else:
                nr_levels_covered = level
        levels = {k: levels[k] for k in range(0,nr_levels_covered + 1)}
        for patch in G.nodes():
            coords_to_extract.extend(boxes[patch])
        
        for level in levels:
            intersection_coords = extract_coords(levels[level], boxes)
            coords_to_extract.extend(intersection_coords)
            missing_coords = extract_coords(levels[level], boxes)
            coords_to_extract.extend(missing_coords)
        a = np.array(coords_to_extract)
        unique_coords = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
        same_coords = 0
        final_coords = []
        for coord in unique_coords:
            if coord[1] == coord[3] or coord[0]==coord[2]:
                same_coords += 1
            else:
                final_coords.append(coord)
            
        
        for coord in final_coords:
            cropped = img[coord[1]:coord[3], coord[0]:coord[2]]
            img_2 = imresize(cropped,(224,224))
    
            train_X1 = np.zeros((1,3,224,224))
            
            if type(img_2[0][0]) == np.dtype('uint8'):
                train_X1[:,0,:,:] = img_2[:,:]
            
                train_X1[:,1,:,:] = img_2[:,:]
            
                train_X1[:,2,:,:] = img_2[:,:]
            else:
                train_X1[:,0,:,:] = img_2[:,:,0]
            
                train_X1[:,1,:,:] = img_2[:,:,1]
            
                train_X1[:,2,:,:] = img_2[:,:,2]
    
            if index_feat % 32 == 0:
                #feat_last_conv = features_caffe_last_conv8(np.array(batch_feature,dtype=np.float32))
                feat_full = features_caffe32(np.array(batch_feature,dtype=np.float32))
                batch_feature = []
                batch_feature.extend(train_X1)
                feat_full_all.extend(feat_full)
                #feat_last_conv_all.extend(feat_last_conv)
            else:
                batch_feature.extend(train_X1)
            index_feat += 1
            
        if len(batch_feature) > 0:
            for cc in batch_feature:
    
                #feat_last_conv = features_caffe_last_conv(np.array(cc,dtype=np.float32))
                feat_full = features_caffe(np.array([cc],dtype=np.float32))
                #feat_last_conv_all.append(feat_last_conv)
                feat_full_all.extend(feat_full)
        print len(feat_full_all)
        print same_coords
        print index_feat
        np.savetxt('/var/node436/local/tstahl/new_Resnet_features/2nd/%s-%s.csv'%(a_i,format(img_nr, "06d")), np.array(feat_full_all), delimiter=",")
        np.savetxt('/var/node436/local/tstahl/new_Resnet_features/2nd/coords/%s-%s.csv'%(a_i,format(img_nr, "06d")), np.array(np.array(final_coords)), delimiter=",")
    
end = time.time()
print end - start

