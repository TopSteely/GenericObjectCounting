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
import get_overlap_ratio
import itertools
from get_intersection import get_intersection
from collections import deque
from itertools import chain, islice
from get_intersection_count import get_intersection_count
#from count_per_lvl import iep,sums_of_all_cliques,count_per_level
import matplotlib.colors as colors
from load import get_seperation, get_data,get_image_numbers,get_class_data, get_traineval_seperation, get_data_from_img_nr
import matplotlib.image as mpimg
from utils import create_tree, find_children, sort_boxes, surface_area, extract_coords, get_set_intersection
from ml import tree_level_regression, tree_level_loss, count_per_level, sums_of_all_cliques, constrained_regression, learn_root, loss, iep_single_patch, iep_single_patch_inverse
import time
import matplotlib.cm as cmx
from scipy import optimize
import time
import cProfile
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib.patches import Rectangle
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from scipy.misc import imread

#class_ = 'sheep'
baseline = False
add_window_size = False
iterations = 1000
subsampling = False
c = 'partial'
normalize = True
prune = False
delta = math.pow(10,-3)
features_used = 5
less_features = False
learn_intersections = True
squared_hinge_loss = False
prune_fully_covered = True
prune_tree_levels = 2
jans_idea = True


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
    X_p, y_p, inv = get_data_from_img_nr(class_,img_nr, subsamples)
    if X_p != []:
        boxes = []
        ground_truth = inv[0][2]
        img_nr = inv[0][0]
        print img_nr
        if less_features:
            X_p = [fts[0:features_used] for fts in X_p]
        if os.path.isfile('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt'):
            f = open('/var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt', 'r')
        else:
            print 'warning, no /var/node436/local/tstahl/Coords_prop_windows/'+ (format(img_nr, "06d")) +'.txt'
        for line, y in zip(f, inv):
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append(coord)
        #assert(len(boxes)<500)
        boxes, y_p, X_p = sort_boxes(boxes, y_p, X_p, 0,5000)
        
        if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
            gr = open('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), 'r')
        else:
            gr = []
        ground_truths = []
        for line in gr:
           tmp = line.split(',')
           ground_truth = []
           for s in tmp:
              ground_truth.append(int(s))
           ground_truths.append(ground_truth)
        
        #prune boxes
        pruned_x = []
        pruned_y = []
        pruned_boxes = []
        if prune:
            for i, y_ in enumerate(y_p):
                if y_ > 0:
                    pruned_x.append(X_p[i])
                    pruned_y.append(y_p[i])
                    pruned_boxes.append(boxes[i])
        else:
            pruned_x = X_p
            pruned_y = y_p
            pruned_boxes = boxes
        
        if subsampling and pruned_boxes > subsamples:
            pruned_x = pruned_x[0:subsamples]
            pruned_y = pruned_y[0:subsamples]
            pruned_boxes = pruned_boxes[0:subsamples]
            
            
        # create_tree
        G, levels = create_tree(pruned_boxes)
        
        #prune tree to only have levels which fully cover the image, tested
        if prune_fully_covered:
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
            
        # prune levels, speedup + performance 
        levels_tmp = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
        levels_gone = {k:v for k,v in levels.iteritems() if k>=prune_tree_levels}
        levels = levels_tmp
        #prune tree as well, for patches training
        for trash_level in levels_gone.values():
            G.remove_nodes_from(trash_level)
        
        coords = []
        features = []
        f_c = []
        f = []
        
        #either subsampling or prune_fully_covered
        #assert(subsampling != prune_fully_covered)
        
        if subsampling:
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r+')
            else:
                if mode == 'extract_train' or mode == 'extract_test':                
                    print 'coords for %s with %s samples have to be extracted'%(img_nr,subsamples)
                    f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'w')
                    for level in levels:
                        levl_boxes = extract_coords(levels[level], pruned_boxes)
                        if levl_boxes != []:
                            for lvl_box in levl_boxes:
                                if lvl_box not in coords:
                                    coords.append(lvl_box)
                                    f_c.write('%s,%s,%s,%s'%(lvl_box[0],lvl_box[1],lvl_box[2],lvl_box[3]))
                                    f_c.write('\n')
                    f_c.close()
                    print 'features for %s with %s samples have to be extracted'%(img_nr,subsamples)
                    os.system('export PATH=$PATH:/home/koelma/impala/lib/x86_64-linux-gcc')
                    os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/koelma/impala/third.13.03/x86_64-linux/lib')
                    #print "EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_%s.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),subsamples,format(img_nr, "06d"),subsamples)
                    os.system("EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/%s_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt"%(class_,(format(img_nr, "06d")),format(img_nr, "06d"),subsamples,class_,format(img_nr, "06d"),subsamples))
                    if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                        f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r')
                    else:
                        f_c = []
            coords = []
                
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/Features_upper/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples)):
                f = open('/var/node436/local/tstahl/Features_prop_windows/Features_upper/%s_%s_%s.txt'%(class_,format(img_nr, "06d"),subsamples), 'r') 
                
                
        elif prune_fully_covered:
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r+')
                
                
            else:
                if mode == 'extract_train' or mode == 'extract_test':                
                    print 'coords for %s with fully_cover_tree samples have to be extracted'%(img_nr)
                    f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'w')
                    for level in levels:
                        levl_boxes = extract_coords(levels[level], pruned_boxes)
                        if levl_boxes != []:
                            for lvl_box in levl_boxes:
                                if lvl_box not in coords:
                                    coords.append(lvl_box)
                                    f_c.write('%s,%s,%s,%s'%(lvl_box[0],lvl_box[1],lvl_box[2],lvl_box[3]))
                                    f_c.write('\n')
                    f_c.close()
                    print 'features for %s with fully_cover_tree samples have to be extracted'%(img_nr)
                    os.system('export PATH=$PATH:/home/koelma/impala/lib/x86_64-linux-gcc')
                    os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/koelma/impala/third.13.03/x86_64-linux/lib')
                    #print "EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_%s.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_%s.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),subsamples,format(img_nr, "06d"),subsamples)
                    print "EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),format(img_nr, "06d"))
                    os.system("EuVisual /var/node436/local/tstahl/Images/%s.jpg /var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt --eudata /home/koelma/EuDataBig --imageroifile /var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt"%((format(img_nr, "06d")),format(img_nr, "06d"),format(img_nr, "06d")))
                    if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                        f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r')
                    else:
                        f_c = []
            coords = []
                
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d"))):
                f = open('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep_%s_fully_cover_tree.txt'%(format(img_nr, "06d")), 'r') 
                        
                
        else:
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep%s.txt'%(format(img_nr, "06d"))):
                f = open('/var/node436/local/tstahl/Features_prop_windows/Features_upper/sheep%s.txt'%(format(img_nr, "06d")), 'r') 
            if os.path.isfile('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep%s.txt'%(format(img_nr, "06d"))):
                f_c = open('/var/node436/local/tstahl/Features_prop_windows/upper_levels/sheep%s.txt'%(format(img_nr, "06d")), 'r+')
                
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
        #assert len(coords) == len(features)
        
        # append x,y of intersections
        if learn_intersections:
            for inters,coord in zip(features,coords):
#                if inters not in pruned_x:
                pruned_x.append(inters)
                ol = 0.0
                ol = get_intersection_count(coord, ground_truths)
                pruned_y.append(ol)
                
        if mode == 'mean_variance':
            print 'normalizing'
            scaler.partial_fit(pruned_x)  # Don't cheat - fit only on training data
            return scaler
            
        if less_features:
            features = [fts[0:features_used] for fts in features]
        #normalize
        norm_x = []
        if normalize and (mode != 'extract_train' and mode != 'extract_test'):
#            for p_x in pruned_x:
#                norm_x.append((p_x-mean)/variance)
            norm_x = scaler.transform(pruned_x)
            if features != []:
                features = scaler.transform(features)
        else:
            norm_x = pruned_x
        data = (G, levels, pruned_y, norm_x, pruned_boxes, ground_truths, alphas)
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
        if mode == 'training':
            if alphas[0] == 0: #if we don't learn the proposals, we learn just the levels: better, because every level has same importance and faster
                w_levels_img=np.zeros(4096,np.dtype('float64'))
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
                    w_level, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,None,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,jans_idea)
                    #print count_per_level([],class_,features,coords,scaler,w_level, np.dot(w,np.array(norm_x).T), img_nr, pruned_boxes,levels[level], '',function)[0]
                    w_levels_img += w_level
                    if level not in functions[img_nr]:
                        functions[img_nr][level] = function
                w_levels_img = w_levels_img / len(levels)
                return w_levels_img, len(pruned_y), len(levels), pruned_y[0]
            else: #if we learn proposals, levels with more proposals have more significance...., slow - need to change
                nodes = list(G.nodes())
                for node in nodes:
                    if node == 0:
                        w = learn_root(w,norm_x[0],pruned_y[0],learning_rate,alphas)
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
                        w, function = constrained_regression(class_,function,features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,squared_hinge_loss)
                        #TODO: train regressor/classifier that predicts/chooses level. Features: level, number of proposals, number of intersections, avg size of proposal, predictions(for regressor), etc.
                        if level not in functions[img_nr]:
                            functions[img_nr][level] = function
                return w, len(pruned_y), len(G.nodes()), pruned_y[0]
        elif mode == 'scikit_train':
            clf.partial_fit(norm_x,pruned_y)
            return clf
        elif mode == 'loss_train':
            if alphas[0] == 0: #levels
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
                return loss__
            else:
                loss__.append(loss(class_,squared_hinge_loss,features,coords,scaler,w, data, predecs, children,img_nr, -1))
        elif mode == 'loss_test' or mode == 'loss_eval':
            if alphas[0] == 0: #levels
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
                cpl = max(0, np.dot(w,np.array(norm_x[0]).T))
                full_image.append([pruned_y[0],cpl])
                return loss__,full_image
            else:
                loss__.append(loss(class_,squared_hinge_loss,features,coords,scaler,w, data, predecs, children,img_nr, -1))
                cpl = max(0, np.dot(w,np.array(norm_x[0]).T))
                full_image.append([pruned_y[0],cpl])
                return loss__,full_image
        elif mode == 'loss_scikit_test' or mode == 'loss_scikit_train':
            loss__.append(((clf.predict(norm_x) - pruned_y)**2).sum())
            return loss__ 
        elif mode == 'levels_train' or mode == 'levels_test':
            preds = []
            for i,x_ in enumerate(norm_x):
                preds.append(np.dot(w, x_))
            cpls = []
            truelvls = []
            used_boxes_ = []
            total_size = surface_area(pruned_boxes, levels[0])
            fully_covered_score = 0.0
            fully_covered_score_lvls = 0.0
            covered_levels = []
            print mode, len(levels)
            best = []
            best_iep=[]
            all_patches = []
            for level in levels:
                iep_boxes_levels_inverse,f = iep_single_patch(y,class_,w,preds,levels,level,features,coords,scaler, norm_x,img_nr, boxes, features, [], jans_idea)
                best_in_level = preds.index(max([preds[l] for l in levels[level]]))
                ind_best_iep_in_level = levels[level][iep_boxes_levels_inverse.index(max(iep_boxes_levels_inverse))]
                best_iep_in_level = [max(iep_boxes_levels_inverse), ind_best_iep_in_level]
                print len(pruned_boxes), best_in_level
                best.append([pruned_boxes[best_in_level], preds[best_in_level]])
                best_iep.append([pruned_boxes[best_iep_in_level[1]], best_iep_in_level[0],pruned_boxes[ind_best_iep_in_level]])
                print pruned_boxes[best_iep_in_level[1]]
                for bb in levels[level]:
                    all_patches.append([pruned_boxes[bb],preds[bb]])
                
                if img_nr in functions:
                    if level in functions[img_nr]:
                        function = functions[img_nr][level]
                    else:
                        function = []
                else:
                    functions[img_nr] = {}
                    function = []
                cpl,used_boxes,_ = count_per_level([],class_,features,coords,scaler,w, preds, img_nr, pruned_boxes,levels[level], '',function)
                # clipp negative predictions
                cpl = max(0,cpl)
                tru = y_p[0]
                cpls.append(cpl)
                sa = surface_area(pruned_boxes, levels[level])
                sa_co = sa/total_size
                if sa_co == 1.0:
                   fully_covered_score += cpl
                   fully_covered_score_lvls += 1
                   covered_levels.append(cpl)
                truelvls.append(tru)
            print best_iep
            return cpls,truelvls,best, all_patches, best_iep
        
            
def main():
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

    gamma = 0.005
    #subsamples_ = [5,8,12]
    if subsampling:
        subsamples = 5
    else:
        subsamples = 100000
    learning_rates = [math.pow(10,-4)]
    learning_rates_ = {}
    if less_features:
        weights_sample = random.sample(range(features_used), 2)
    else:
        weights_sample = random.sample(range(4096), 10)
    all_alphas = [1]
    regs = [1e-6]
    n_samples = 0.0
    if less_features:
        sum_x = np.zeros(features_used)
        sum_sq_x = np.zeros(features_used)
    else:
        sum_x = np.zeros(4096)
        sum_sq_x = np.zeros(4096)
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
    functions = {}

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
    if normalize:
        print 'normalizing scaler'
        scaler = MinMaxScaler()
        for img_nr in tr_images:
            print img_nr
            
            scaler = minibatch_(None,None,scaler,[], [],[],[],[],[],img_nr,[],[],subsamples,'mean_variance')
     
    #normalize
    learning_rate0 = learning_rates[0]
    learning_rate = learning_rate0
    alpha1 =         all_alphas[0]
    reg = regs[0]
    alphas_levels = [0,1,reg]
    alphas_patches = [1,reg,1,1]
    alphas_just_patches = [1, reg, 0, 0]
    alphas_just_parent = [1, reg, 1, 0]
    alphas_just_level = [1, reg, 0, 1]
    X_ = []
    y_ = []
    X_test = []
    y_test = []
    w_all = {}
    levels_num = 8
    t = 0
    for epochs in np.arange(6):
        global prune_tree_levels
        prune_tree_levels = levels_num
        alphas = [1-alpha1,alpha1,reg]
        
        if epochs == 0:
            # initialize or reset w , plot_losses
            w_all[levels_num] = []
            if less_features:
                w_levels = 0.01 * np.random.rand(features_used)
                w_patches = 0.01 * np.random.rand(features_used)
            else:
                w_levels = 0.01 * np.random.rand(4096)     
                w_patches = 0.01 * np.random.rand(4096)     
                w_just_patches = 0.01 * np.random.rand(4096)
                w_just_parent = 0.01 * np.random.rand(4096)
                w_just_level = 0.01 * np.random.rand(4096)
                w_new_loss = 0.01 * np.random.rand(4096)
            plot_training_loss_levels = []
            plot_evaluation_loss_levels = []
            plot_training_loss_patches = []
            plot_evaluation_loss_patches = []
        loss_train= []
        loss_test = []
        full_image_test = []
        full_image_train = []
        learning_rate_again = []
        start = time.time()    
        
        mse_train_ = []
        mse_test_ = []
        
        mse_mxlvl_train = []
        mse_mxlvl_test = []
        
        mse_fllycover_train = []
        mse_fllycover_test = []
        
        clte = []
        cltr = []
        
        lmte = []
        lmtr = []
        new = True
        print 'training model'
        loss_train_levels = []
        loss_eval_levels = []
        loss_train_patches = []
        loss_eval_patches = []
        full_image__train = []
        print epochs, learning_rate, alphas
        #shuffle images, not boxes!
        y_train = []
        for img_nr in tr_images:
            w_temp,le,nr_nodes, y = minibatch_(functions,None,scaler,w_levels, [],[],[],[],[],img_nr,alphas_levels,learning_rate,subsamples,'training')
            w_levels -= (learning_rate * w_temp)
            t += nr_nodes
            y_train.append(y)
        learning_rate = learning_rate0 * (1+learning_rate0*gamma*t)**-1
                    
                    
    plt.figure()
    for name, w, alpha in zip(['levels'],[w_levels],[alphas_levels]):
        
        predictions = {}
        over_under = {}
        distance = {}
        levels_c = {}
        levels_error = []
        patches_error = []

        for img_nr in te_images:
            cpls,trew,best,all_patches, best_iep = minibatch_(functions, [],scaler,w, [],[],[],[],[],img_nr,alphas,learning_rate0,subsamples, 'levels_test')
            if len(cpls) < 6 or trew[0] < 2:
                continue
            im = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
            for p in all_patches:
                patches_error.append((p[0][1]-p[1])**2)
            for i,(b,c,b_iep) in enumerate(zip(best,cpls,best_iep)):
                levels_error.append((c-trew[0])**2)
                coord_iep = b_iep[0]
                plt.imshow(im)
                plt.axis('off')
                ax = plt.gca()
                #ax.add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), edgecolor='black', facecolor='none'))
                ax.add_patch(Rectangle((int(coord_iep[0]), int(coord_iep[1])), int(coord_iep[2] - coord_iep[0]), int(coord_iep[3] - coord_iep[1]), edgecolor='red', facecolor='none'))
                ax.set_title('IEP best: %s\n IEP Level: %s'%(b_iep[1],c))
                
            
                plt.savefig('/home/tstahl/best/best_preds_%s_%s.png'%(img_nr,i))
                plt.clf()
        print name, 'levels error: ',np.array(levels_error).sum()/len(levels_error), 'patches error: ',np.array(patches_error).sum()/len(patches_error), 
def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3]), 1/((A[2]- A[0] + 1)*(A[3]-A[1] + 1))
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
