# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:20:18 2015

@author: root
"""

from sklearn import linear_model, preprocessing
import matplotlib
matplotlib.use('agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy
import math
import sys
import random
import pylab as pl
import networkx as nx
import pyximport; pyximport.install(pyimport = True)
#import get_overlap_ratio
import itertools
from get_intersection import get_intersection
from collections import deque
from itertools import chain, islice
from get_intersection_count import get_intersection_count
#from count_per_lvl import iep,sums_of_all_cliques,count_per_level
import matplotlib.colors as colors
from load import get_seperation, get_data,get_image_numbers,get_class_data, get_traineval_seperation, get_data_from_img_nr,get_grid_resnet_data_from_img_nr,get_features_new, get_grid_data_from_img_nr, get_features, get_all_grids_resnet_data_from_img_nr, get_all_grid_boxes
import matplotlib.image as mpimg
from utils import create_tree, find_children, sort_boxes, surface_area, extract_coords, get_set_intersection, extract_coords_missing
from ml import tree_level_regression_new_features, tree_level_loss, count_per_level_new, sums_of_all_cliques, constrained_regression_new_features, learn_root, loss, learn_grid, tree_level_regression, constrained_regression, count_per_level, iep_single_patch_new
import time
import matplotlib.cm as cmx
from scipy import optimize
import time
import cProfile
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib.patches import Rectangle
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from random import shuffle
from scipy.misc import imread


#class_ = 'sheep'
baseline = False
add_window_size = False
iterations = 1000
c = 'partial'
normalize = True
delta = math.pow(10,-3)
features_used = 5
less_features = False
learn_intersections = False
squared_hinge_loss = False
prune_fully_covered = True
prune_tree_levels = 2
jans_idea = True
new_coords = True


def get_labels(class_,i, criteria, subsamples):
    labels = []
    if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt'):
        file = open('/var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt', 'r')
    else:
        print 'warning /var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/'+(format(i, "06d")) + '_' + class_ + '_' + criteria + '.txt does not exist '
        return np.zeros(subsamples)
    for i_l, line in enumerate(file):
        tmp = line.split()[0]
        labels.append(float(tmp))
        if i_l == subsamples - 1:
            break
    return labels


def minibatch_(functions, clf,scaler,w, loss__,mse,hinge1,hinge2,full_image,img_nr,alphas,learning_rate,subsamples, mode):
    #print img_nr
    if mode.startswith('grid'):
        if new_coords:
            X_p = get_all_grids_resnet_data_from_img_nr(img_nr)
        else:
            if os.path.isfile('/var/node436/local/tstahl/old_hard_features/'+ (format(img_nr, "06d")) +'3x3.txt'):
                f = open('/var/node436/local/tstahl/old_hard_features/'+ (format(img_nr, "06d")) +'3x3.txt', 'r')
            else:
                print 'warning no /var/node436/local/tstahl/old_hard_features', img_nr
            X_p = []
            for i_n, line in enumerate(f):
                tmp = line.split(',')
                coord = []
                for s in tmp:
                    coord.append(float(s))
                X_p.append(coord)
                if i_n == subsamples - 1:
                    break
            
    else:
        if new_coords:
            X_p = get_features_new(img_nr, subsamples)
        else:
            X_p = get_features(img_nr, subsamples)
    if mode == 'mean_variance' or mode == 'grid_mean_variance':
        scaler.partial_fit(X_p)
        return scaler
    y_p = get_labels(class_,img_nr, 'partial', subsamples)
    ground_truth = y_p[0]

    if X_p != []:
        boxes = []
        
        if mode.startswith('grid'):
            #boxes = get_all_grid_boxes(img_nr)
            'a'
        else:
            if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt'):
                f = open('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt', 'r')
            else:
                print 'warning, no /var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt'
            for line in f:
                tmp = line.split(',')
                coord = []
                for s in tmp:
                    coord.append(float(s))
                boxes.append(coord)
            if new_coords:
                boxes, y_p, _ = sort_boxes(boxes, y_p, np.zeros(5000), 0,5000)
            else:
                boxes, y_p, X_p = sort_boxes(boxes, y_p, X_p, 0,5000)
        
        pruned_x = X_p
        pruned_y = y_p
        pruned_boxes = boxes
            
            
        # create_tree
        if not mode.startswith('grid'):
            all_coords = []
            unique_coords = []
            lut = {}
            # after extraction of coords, create tree and extract needed intersections
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
        
                      
            if new_coords:
                #have to check if levels boxes are in same order after unique
                test = []
                for patch in G.nodes():
                    if pruned_boxes[patch] == []:
                        print 'pruned_boxes[patch] == [] ', patch, pruned_boxes[patch], len(pruned_boxes)
                    all_coords.append(pruned_boxes[patch])
                
                for level in levels:
                    intersection_coords = extract_coords(levels[level], pruned_boxes)
                    test.extend([pruned_boxes[ll] for ll in levels[level]])
                    #for i_coo in intersection_coords:
                    all_coords.extend(intersection_coords)
                a = np.array(all_coords)
                #print a
                unique_coords = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
                unique_coords_copy = unique_coords
                #print unique_coords
                #raw_input()
                same_coords = 0
                to_delete = []
                for t_d,coord in enumerate(unique_coords):
                    if coord[1] == coord[3] or coord[0]==coord[2]:
                        same_coords += 1
                        to_delete.append(t_d)
                        
                #for t_d in reversed(to_delete):
                deleted = 0
                for t_d in to_delete:
                    unique_coords = np.delete(unique_coords,t_d-deleted,0)
                    deleted += 1
                    
                for t_d in reversed(to_delete):
                    unique_coords_copy = np.delete(unique_coords_copy,t_d,0)
                assert (np.array_equal(unique_coords,unique_coords_copy))
                # reorder G,levels,y to match unique_coords
                y_temp = np.ones(len(unique_coords)) * -1 # just to debug, should never access any y that is -1
    
                levels_reordered = {}
                G_reordered = nx.Graph()
                parents = nx.dfs_predecessors(G)
                for level in levels:
                    levels_reordered[level] = []
                    if level == 0:
                        new_index = unique_coords.tolist().index(pruned_boxes[0])
                        levels_reordered[level].append(new_index)
                        y_temp[new_index] = pruned_y[0]
                        G_reordered.add_node(new_index)
                        lut[new_index] = 0
                        continue
                    for patch in levels[level]:
    
                        parent = parents[patch]
                        parent_new = unique_coords.tolist().index(pruned_boxes[parent])
                        new_index = unique_coords.tolist().index(pruned_boxes[patch])
    
                        levels_reordered[level].append(new_index)
                        G_reordered.add_edge(parent_new,new_index)
                        y_temp[new_index] = pruned_y[patch]
                        lut[new_index] = patch
                        
                test1 = []
                for level in levels_reordered:
                    test1.extend([unique_coords[ll] for ll in levels_reordered[level]])
    
                missing_coords = []
                for level in levels:
                    mcoords = extract_coords_missing(levels[level], pruned_boxes)
                    missing_coords.extend(mcoords)
                if missing_coords != []:
                    a = np.array(missing_coords)
                    unique_missing_coords = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
            
                    unique_coords = np.concatenate((unique_coords, unique_missing_coords), axis=0)
                #test if reordering is correct
                assert np.array_equal(test,test1)
                
               # print len(unique_coords), len(pruned_x)
                #just one level - full image
                if len(unique_coords) != 1:
                    #assert (len(unique_coords) == len(pruned_x))
                    0
                else:
                    pruned_x = [pruned_x]
    
                del pruned_boxes
                levels = levels_reordered
                G = G_reordered
                pruned_y = y_temp
            else:
                if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                    f = open('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r') 
                if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                    f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r')
                else:
                    print '/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))
                
                coords = []
                features = []
                if f_c != []:
                    for i,line in enumerate(f_c):
                        str_ = line.rstrip('\n').split(',')
                        cc = []
                        for s in str_:
                           cc.append(float(s))
                        coords.append(cc)
                if f != []:
                    for i,line in enumerate(f):
                        str_ = line.rstrip('\n').split(',')  
                        ff = []
                        for s in str_:
                           ff.append(float(s))
                        features.append(ff)
             # prune levels, speedup + performance 
            levels_tmp = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
            levels_gone = {k:v for k,v in levels.iteritems() if k>=prune_tree_levels}
            levels = levels_tmp
            #prune tree as well, for patches training
            for trash_level in levels_gone.values():
                G.remove_nodes_from(trash_level)
            
        #normalize
        norm_x = []
        if normalize:
#            for p_x in pruned_x:
#                norm_x.append((p_x-mean)/variance)
            norm_x = scaler.transform(pruned_x)
        else:
            norm_x = pruned_x

        if not mode.startswith('grid'):
            if not new_coords:
                if features != []:
                    features = scaler.transform(features)
            sucs = nx.dfs_successors(G)
            
            predecs = nx.dfs_predecessors(G)
            
            #preprocess: node - children
            children = {}
            last = -1
            for node,children_ in zip(sucs.keys(),sucs.values()):
                if node != last+1:
                    for i in range(last+1,node):
                        children[i] = []
                    children[node] = children_
                elif node == last +1:
                    children[node] = children_
                last = node
        if mode.startswith('training') and mode != 'training_error':
            if alphas[0] == 0: #if we don't learn the proposals, we learn just the levels: better, because every level has same importance and faster
                w_levels_img=np.zeros(len(w),np.dtype('float64'))
                level_preds = []
                for level in levels:
                    if img_nr in functions:
                        if level in functions[img_nr]:
                            function = functions[img_nr][level]
                        else:
                            function = []
                    else:
                        functions[img_nr] = {}
                        function = []
                    #print count_per_level([],class_,features,coords,scaler,w, np.dot(w,np.array(norm_x).T), img_nr, pruned_boxes,levels[level], '',function)[0]
                    if new_coords:
                        if mode == 'training_max':
                            cpl = count_per_level_new(class_,norm_x,unique_coords,scaler,w, img_nr, levels[level], '',function)
                            level_preds.append(cpl)
                        elif mode == 'training_average':
                            w_tmp, function = tree_level_regression_new_features(class_,function,levels,level,unique_coords,scaler,w,norm_x,ground_truth,alphas,img_nr)
                            w_levels_img += w_tmp
                        else:
                            print 'wrong name'
                            exit()
                    else:
                        if mode == 'training_max':   
                            preds = np.dot(w,np.array(norm_x).T)
                            cpl,_,_ = count_per_level([],class_,features,coords,scaler,w,preds,img_nr, pruned_boxes,  levels[level], '', function)
                            level_preds.append(cpl)
                        elif mode == 'training_average':
                            w_tmp, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,None,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,jans_idea)
                            w_levels_img += w_tmp
                        else:
                            print 'wrong name'
                            exit()
                    #print count_per_level([],class_,features,coords,scaler,w_level, np.dot(w,np.array(norm_x).T), img_nr, pruned_boxes,levels[level], '',function)[0]
                    #w_levels_img += w_level
                    if level not in functions[img_nr]:
                        functions[img_nr][level] = function
                if mode == 'training_average':
                    w_ret = w_levels_img / len(levels)
                elif mode == 'training_max':
                    ind_max = level_preds.index(max(level_preds))
                    if new_coords:
                        w_ret, function = tree_level_regression_new_features(class_,function,levels,ind_max,unique_coords,scaler,w,norm_x,ground_truth,alphas,img_nr)
                    else:
                        w_ret, function = tree_level_regression(class_,function,levels,ind_max,features,coords,scaler,w,norm_x,pruned_y,None,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,jans_idea)
                else:
                    print 'wrong name'
                    exit()
                return w_ret, len(pruned_y), len(levels), ground_truth
            else: #if we learn proposals, levels with more proposals have more significance...., slow - need to change
                nodes = list(G.nodes())
                w_update = np.zeros(len(w))
                for node in nodes:
                    if node == G.nodes()[0]:
                        w_update += learn_root(w,norm_x[node],ground_truth,learning_rate,alphas)
                    else:
                        for num,n in enumerate(levels.values()):
                            if node in n:
                                level = num
                                break
                        if img_nr in functions:
                            if level in functions[img_nr]:
                                function = functions[img_nr][level]
                            else:
                                function = []
                        else:
                            functions[img_nr] = {}
                            function = []
                        #w, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr)
                        if new_coords:
                            w_update += constrained_regression_new_features(class_,unique_coords,scaler,w,norm_x,pruned_y,node,predecs,children,learning_rate,alphas,img_nr)
                        else:
                            w_update += constrained_regression(class_,features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,squared_hinge_loss)
                        #TODO: train regressor/classifier that predicts/chooses level. Features: level, number of proposals, number of intersections, avg size of proposal, predictions(for regressor), etc.
                        if level not in functions[img_nr]:
                            functions[img_nr][level] = function
                w_update = w_update / len(G.nodes())
                return w_update, len(pruned_y), len(G.nodes()), ground_truth
        elif mode.startswith('gridunstrain'):
            grid_levels = {}
            grid_levels[0] = [norm_x[0]]
            grid_levels[1] = norm_x[1:3]
            grid_levels[2] = norm_x[3:7]
            grid_levels[3] = norm_x[7:16]
            grid_levels[4] = norm_x[16:32]            
            if mode == 'gridunstrain_max':
                levels_preds = []
                levels_preds.append(np.dot(w,norm_x[0]))
                level1 = 0.0
                for x_ in norm_x[1:3]:
                    level1 += (np.dot(w,x_))
                levels_preds.append(level1)
                level2 = 0.0
                for x_ in norm_x[3:7]:
                    level2 += (np.dot(w,x_))
                levels_preds.append(level2)
                level3 = 0.0
                for x_ in norm_x[7:16]:
                    level3 += (np.dot(w,x_))
                levels_preds.append(level3)
                level4 = 0.0
                for x_ in norm_x[16:32]:
                    level4 += (np.dot(w,x_))
                levels_preds.append(level4)
                ind_max = levels_preds.index(max(levels_preds))
                return learn_grid(w,grid_levels[ind_max],ground_truth,learning_rate)
            elif mode == 'gridunstrain_average':
                for gr_lvl in grid_levels:
                    w_update = np.zeros(len(w))
                    w_update += learn_grid(w,grid_levels[gr_lvl],ground_truth,learning_rate)
                return w_update / 4.0
            else:
                print 'wrong name'
                exit()
            return w
                
        elif mode.startswith('grid_unsup'):
            grid_levels = {}
            grid_levels[0] = [norm_x[0]]
            grid_levels[1] = norm_x[1:3]
            grid_levels[2] = norm_x[3:7]
            grid_levels[3] = norm_x[7:16]
            grid_levels[4] = norm_x[16:32]
            lvl_preds = []
            for gr_lvl in grid_levels:
                lvl_pred = 0.0
                for x_ in grid_levels[gr_lvl]:
                    lvl_pred += (np.dot(w,x_))
                lvl_preds.append(lvl_pred)
            if mode == 'grid_unsup_max':
                return np.max(lvl_preds)
            elif mode == 'grid_unsup_median':
                return np.median(lvl_preds)
            else:
                print 'wrong name'
                exit()
        else:
            preds = []
            level_preds = []
            for gi in G.nodes():
                preds.append(np.dot(w, norm_x[gi]))
            for level in levels:
                level_patches_pred = 0.0
                for ni in levels[level]:
                    level_patches_pred += np.dot(w, norm_x[ni])
                level_preds.append(level_patches_pred)
            cpls = []
            truelvls = []
            for level in levels:
                if img_nr in functions:
                    if level in functions[img_nr]:
                        function = functions[img_nr][level]
                    else:
                        function = []
                else:
                    functions[img_nr] = {}
                    function = []
                if new_coords:
                    cpl = count_per_level_new(class_,norm_x,unique_coords,scaler,w, img_nr, levels[level], '',function)
                else:
                    preds_tmp = np.dot(w,np.array(norm_x).T)
                    cpl,_,_ = count_per_level([],class_,features,coords,scaler,w, preds_tmp, img_nr, pruned_boxes,levels[level], '',function)
                # clipp negative predictions
                cpl = max(0,cpl)
                tru = ground_truth
                cpls.append(cpl)
                truelvls.append(tru)
            if mode == 'full_image':
                #print cpl
                #assert isinstance(cpl, numpy.float64)
                return cpl
            elif mode == 'levels_median':
                return np.median(cpls),((np.array(cpls)-tru)**2).sum()
            elif mode == 'levels_max':
                return np.max(cpls)
            elif mode == 'just_patches':
                return np.mean(cpls),preds, level_preds
            elif mode == 'training_error':
                return ((np.array(cpls)-tru)**2).sum()
            elif mode == 'show_best':
                preds = []
                for i,x_ in enumerate(norm_x):
                    preds.append(np.dot(w, x_))
                cpls = []
                truelvls = []
                used_boxes_ = []
                total_size = surface_area(unique_coords, levels[0])
                fully_covered_score = 0.0
                fully_covered_score_lvls = 0.0
                covered_levels = []
                print mode, len(levels)
                best = []
                best_iep=[]
                for level in levels:
                    iep_boxes_levels_inverse,f = iep_single_patch_new(unique_coords,ground_truth,class_,w,levels,level,scaler, norm_x,img_nr,function)
                    best_in_level = preds.index(max([preds[l] for l in levels[level]]))
                    ind_best_iep_in_level = levels[level][iep_boxes_levels_inverse.index(max(iep_boxes_levels_inverse))]
                    best_iep_in_level = [max(iep_boxes_levels_inverse), ind_best_iep_in_level]
                    best.append([unique_coords[best_in_level], preds[best_in_level]])
                    best_iep.append([unique_coords[best_iep_in_level[1]], best_iep_in_level[0],unique_coords[ind_best_iep_in_level]])
                    if img_nr in functions:
                        if level in functions[img_nr]:
                            function = functions[img_nr][level]
                        else:
                            function = []
                    else:
                        functions[img_nr] = {}
                        function = []
                return best, best_iep
        
            else:
                print 'wrong name'
                exit()
            
def main():
    start = time.time()
    test_imgs, train_imgs = get_seperation()
    train_imgs = train_imgs
    test_imgs = test_imgs
    training_imgs, evaluation_imgs = get_traineval_seperation(train_imgs)
    # learn
#    if os.path.isfile('/var/node436/local/tstahl/Models/'+class_+c+'normalized_constrained.pickle'):
#        with open('/var/node436/local/tstahl/Models/'+class_+c+'normalized_constrained.pickle', 'rb') as handle:
#            w = pickle.load(handle)
#    else:
    evaluation_imgs = evaluation_imgs
    img_train = training_imgs
    img_eval = evaluation_imgs
    test_imgs = test_imgs
    gamma = 0.5
    #subsamples_ = [5,8,12]
    subsamples = 100000
    learning_rates = [math.pow(10,-3)]
    learning_rates_ = {}
    all_alphas = [1]
    regs = [1e-6]
    n_samples = 0.0
    if less_features:
        sum_x = np.zeros(features_used)
        sum_sq_x = np.zeros(features_used)
    else:
        sum_x = np.zeros(1000)
        sum_sq_x = np.zeros(1000)
    if len(sys.argv) != 2:
        print 'wrong arguments'
        exit()
    mous = 'whole'
    global class_
    class_ = sys.argv[1]
    
    print 'learning', class_
    if mous != 'whole':
        train_imgs = get_image_numbers(test_imgs,train_imgs,class_)
    plt.figure()
    mean = []
    variance = []
    scaler = []
    #learn regressors
    #want to just learn images with objects present
    tr_images = []
    te_images = []
    for img in img_train:
        y = get_labels(class_,img, 'partial', 1)
        if y[0] > 0:
            tr_images.append(img)
    for img in img_eval:
        y = get_labels(class_,img, 'partial', 1)
        if y[0] > 0:
            te_images.append(img)

    functions = {}
    global new_coords
    new_coords = True
    if normalize:
        #normalize
        print 'normalizing scaler'
        scaler = MinMaxScaler()
        for im_i,img_nr in enumerate(tr_images):
            if img_nr == 3577:
                continue
            y_p = get_labels(class_,img_nr, 'partial', 1)
            if y_p[0] > 0:
                scaler = minibatch_(None,None,scaler,[], [],[],[],[],[],img_nr,[],[],subsamples,'mean_variance')
        
    learning_rate0 = learning_rates[0]
    learning_rate = learning_rate0
    alpha1 =         all_alphas[0]
    reg = regs[0]
    alphas_levels = [0,1,reg]
    alphas_just_patches = [1, reg, 0, 0]
    alphas_ = [1,0,0,0]
    shuffle(tr_images)
    plt.figure()
    mse_level = []
    for learning_rate0 in learning_rates:
        
        for levels_num in [6]:
            train_error = []
            test_error = []
            train_error_epch_after = []
            test_error_epch_after = []
            learning_rate = learning_rate0
            learning_rate1 = learning_rate0
            if new_coords:
                w_levels_median = 0.0001 * np.random.rand(1000)
                w_levels_median1 = 0.0001 * np.random.rand(1000)
                w_update_levels_median = np.zeros(1000,np.dtype('float64'))  
                w_update_levels_median1 = np.zeros(1000,np.dtype('float64'))  
            else:
                w_levels_median = 0.0001 * np.random.rand(4096)
                w_levels_median1 = 0.0001 * np.random.rand(4096)
                w_update_levels_median = np.zeros(4096,np.dtype('float64'))  
                w_update_levels_median1 = np.zeros(4096,np.dtype('float64'))  
                
            t = 0
            for epochs in np.arange(3):
                
                global prune_tree_levels
                prune_tree_levels = levels_num
                        
                        
                    #[w_levels_max, w_levels_median, w_just_patches, w_grid3x3, w_grid3x3_unsup, w_grid_max, w_grid_unsup_max, w_grid_median, w_grid_unsup_median, w_full]
    
    
                print epochs, learning_rate
                #shuffle images, not boxes!
                y_train = []
                
                for m_b,img_nr in enumerate(tr_images):
                    if img_nr == 3577:
                        continue
                    w_temp_levels_median,_,_,_ = minibatch_(functions,None,scaler,w_levels_median, [],[],[],[],[],img_nr,alphas_levels,learning_rate,subsamples,'training_average')
                    w_update_levels_median += w_temp_levels_median
                    #update w after 5 minibatches
                    if m_b % 5 == 4:
                        w_levels_median -= (learning_rate * w_update_levels_median)
                        #variance of whole training set
                        y_train.append(y)
                        t += 5
                        learning_rate = learning_rate * (1+learning_rate0*gamma*t)**-1
                #update remaining images if number of images modulo 5 not == 0
                w_levels_median -= (learning_rate * w_update_levels_median)
                w_update_levels_median = np.zeros(len(w_update_levels_median),np.dtype('float64'))
                t += (m_b%5)
                learning_rate = learning_rate * (1+learning_rate0*gamma*t)**-1
                loss = 0.0
                for m_b,img_nr in enumerate(tr_images):
                    if img_nr == 3577:
                        continue
                    
                
                name = 'levels_median'
                w = w_levels_median
                alphas = alphas_levels
                error = 0.0
                error_1 = 0.0
                err_obj_per_image_ours = {}
                num_per_image = {}
                loss = 0.0
                for te_x,img_nr in enumerate(te_images):
                    best, best_iep = minibatch_(functions, [],scaler,w, [],[],[],[],[],img_nr,alphas,learning_rate0,subsamples, 'show_best')
                    im = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
                    for i,(b,b_iep) in enumerate(zip(best,best_iep)):
                        coord = b[0]
                        coord_iep = b_iep[0]
                        plt.imshow(im)
                        plt.axis('off')
                        ax = plt.gca()
                        ax.add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), edgecolor='black', facecolor='none'))
                        ax.add_patch(Rectangle((int(coord_iep[0]), int(coord_iep[1])), int(coord_iep[2] - coord_iep[0]), int(coord_iep[3] - coord_iep[1]), edgecolor='red', facecolor='none'))
                        ax.set_title('Prediction: %s: %s IEP best: %s'%(b[1], b_iep[1],b_iep[2]))
                        
                    
                        plt.savefig('/home/tstahl/new_best/test_%s_%s_best_preds_%s_best_iep_normal.png'%(img_nr,name,i))
                        plt.clf()
def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3]), 1/((A[2]- A[0] + 1)*(A[3]-A[1] + 1))
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
