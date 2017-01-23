# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:20:18 2015

@author: root
"""

import optunity
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
from load import get_seperation, get_data,get_image_numbers,get_class_data, get_traineval_seperation, get_labels, get_features, get_features_new
import matplotlib.image as mpimg
from utils import create_tree, find_children, sort_boxes, surface_area, extract_coords, get_set_intersection
from ml import tree_level_regression, tree_level_loss, count_per_level, sums_of_all_cliques
import time
import matplotlib.cm as cmx
from scipy import optimize
import time
import cProfile
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib.patches import Rectangle
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
import gzip
import scipy
from scipy.misc import imread
import optunity.metrics

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
classes = ['person','dog','bird','sheep','bus','cat','aeroplane','motorbike','boat','bottle','pottedplant','tvmonitor','cow','horse','bicycle','train','car','chair','diningtable','sofa']


def minibatch_(functions, clf,scaler,w, loss__,mse,hinge1,hinge2,full_image,alphas,learning_rate,test_imgs, train_imgs, img_train, img_eval,minibatch,subsamples,sum_x,n_samples,sum_sq_x,mean,variance, mode):
    print minibatch
    if mode == 'loss_test' or mode == 'loss_scikit_test' or mode == 'levels_test' or mode == 'extract_test':
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, img_train, img_eval, minibatch, minibatch + 1, 'test', c,subsamples)                
    elif mode == 'train':
        X_p, y_p, inv = get_class_data(class_, test_imgs, train_imgs, img_train, img_eval, minibatch, minibatch + 1, 'training', c,subsamples)        
    elif mode == 'training':
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, img_train, img_eval, minibatch, minibatch + 1, 'training', c,subsamples)        
    elif mode == 'loss_eval':
        X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, img_train, img_eval, minibatch, minibatch + 1, 'eval', c,subsamples)        
        
    else:
         X_p, y_p, inv = get_data(class_, test_imgs, train_imgs, img_train, img_eval, minibatch, minibatch + 1, 'training', c,subsamples)    
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
            print 'warning'
        for line, y in zip(f, inv):
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append([coord, y[2]])
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
        levels = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
        
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
            sum_x += np.array(pruned_x).sum(axis=0)
            n_samples += len(pruned_x)
            sum_sq_x +=  (np.array(pruned_x)**2).sum(axis=0)
            scaler.partial_fit(pruned_x)  # Don't cheat - fit only on training data
            return sum_x,n_samples,sum_sq_x, scaler
            
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
        if mode == 'train':
            if alphas[0] == 0: #if we don't learn the proposals, we learn just the levels: better, because every level has same importance and faster
                for level in levels:
                    print 'level' , level
                    if img_nr in functions:
                        if level in functions[img_nr]:
                            function = functions[img_nr][level]
                        else:
                            function = []
                    else:
                        functions[img_nr] = {}
                        function = []
                    w, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,None,predecs,children,pruned_boxes,learning_rate,alphas,img_nr,jans_idea)
                    if level not in functions[img_nr]:
                        functions[img_nr][level] = function
                return w, len(pruned_y), len(levels)
            else: #if we learn proposals, levels with more proposals have more significance...., slow - need to change
                nodes = list(G.nodes())
                for node in nodes:
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
                    w, function = tree_level_regression(class_,function,levels,level,features,coords,scaler,w,norm_x,pruned_y,node,predecs,children,pruned_boxes,learning_rate,alphas,img_nr)
                    #TODO: train regressor/classifier that predicts/chooses level. Features: level, number of proposals, number of intersections, avg size of proposal, predictions(for regressor), etc.
                    if level not in functions[img_nr]:
                        functions[img_nr][level] = function
            return w, len(pruned_y), len(G.nodes())
        elif mode == 'scikit_train':
            clf.partial_fit(norm_x,pruned_y)
            return clf
        elif mode == 'loss_train':
            if clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] == 0:
                loss__.append(0)
            elif clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] > 0:
                loss__.append(pruned_y[0]**2)
            else:
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
            return loss__
        elif mode == 'loss_test' or mode == 'loss_eval':
            if clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] == 0:
                loss__.append(0)
                full_image.append([pruned_y[0],0])
            elif clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] > 0:
                loss__.append(pruned_y[0]**2)
                full_image.append([pruned_y[0],0])
            else:
                loss__.append(tree_level_loss(class_,features,coords,scaler, w, data, predecs, children,img_nr,-1,functions))
                cpl = max(0, np.dot(w,np.array(norm_x[0]).T))
                full_image.append([pruned_y[0],cpl])
            return loss__,full_image
        elif mode == 'loss_scikit_test' or mode == 'loss_scikit_train':
            loss__.append(((clf.predict(norm_x) - pruned_y)**2).sum())
            return loss__ 
        elif mode == 'levels_train' or mode == 'levels_test':
            #im = mpimg.imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
            if clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] == 0:
                cpls = []
                truelvls = []
                for level in levels:
                    cpls.append(0)
                    truelvls.append(0)
            elif clf.predict(np.array(X_p[0]).reshape(1, -1)) == 0 and pruned_y[0] > 0:
                loss__.append(pruned_y[0]**2)
                cpls = []
                truelvls = []
                for level in levels:
                    cpls.append(0)
                    truelvls.append(pruned_y[0])
            else:
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
                for level in levels:
                    function = functions[img_nr][level]
                    cpl,used_boxes,_ = count_per_level([],class_,features,coords,scaler,w, preds, img_nr, pruned_boxes,levels[level], '',function)
                    # clipp negative predictions
                    cpl = max(0,cpl)
                    if used_boxes != []:
                        used_boxes_.append(used_boxes[0][1])
                    tru = y_p[0]
                    cpls.append(cpl)
                    sa = surface_area(pruned_boxes, levels[level])
                    sa_co = sa/total_size
                    if sa_co == 1.0:
                       fully_covered_score += cpl
                       fully_covered_score_lvls += 1
                       covered_levels.append(cpl)
                    truelvls.append(tru)
            return cpls,truelvls
        
            
def main():
    
    test_imgs, train_imgs = get_seperation()
    #train_imgs = train_imgs[0:37]
    #test_imgs = test_imgs[0:37]
    img_train, img_eval = get_traineval_seperation(train_imgs)
    if subsampling:
        subsamples = 5
    else:
        subsamples = 1
    learning_rates = [0.00001]
    learning_rates_ = {}
    if less_features:
        weights_sample = random.sample(range(features_used), 2)
    else:
        weights_sample = random.sample(range(4096), 10)
    all_alphas = [1]
    regs = [0]
    n_samples = 0.0
    if less_features:
        sum_x = np.zeros(features_used)
        sum_sq_x = np.zeros(features_used)
    else:
        sum_x = np.zeros(4096)
        sum_sq_x = np.zeros(4096)
    mous = 'whole'
    mse_classes = {}
    for class_temp in classes:
        global class_
        class_ = class_temp
        if mous != 'whole':
            train_imgs = get_image_numbers(test_imgs,train_imgs,class_)
        plt.figure()
        mean = []
        variance = []
        scaler = []
        functions = {}
        class_images_ = get_image_numbers(test_imgs,train_imgs,class_)
        training_class_images = filter(lambda x:x in train_imgs,class_images_)
    
        scaler_old = MinMaxScaler()
        scaler_new = MinMaxScaler()
        for img_nr in img_train[0:400]:
            if get_labels(class_,img_nr, 'partial', 1)[0] > 0:
                #load image
                #get features
                feat_new = get_features_new(img_nr, 1)
                feat_old = get_features(img_nr, 1)
                scaler_new.partial_fit(feat_new)  
                scaler_old.partial_fit(feat_old) 
        learning_rate0 = learning_rates[0]
        alpha1 =         all_alphas[0]
        reg = regs[0]
        alphas = [1-alpha1,alpha1,reg]

        for learning_rate0 in [math.pow(10,-3),math.pow(10,-4),math.pow(10,-5),math.pow(10,-6)]:
        
            clf_old = linear_model.SGDRegressor(eta0=learning_rate0, learning_rate='invscaling')
            clf_new = linear_model.SGDRegressor(eta0=learning_rate0, learning_rate='invscaling')
            mlp_old = MLPRegressor(verbose=True, hidden_layer_sizes=(250,250), learning_rate='invscaling')
            mlp_new = MLPRegressor(verbose=True, hidden_layer_sizes=(250,250), learning_rate='invscaling')
            
            y_class = []
            X_all_ = []
            X_new_all = []
            for it,img_nr in enumerate(img_train[0:400]):
                y_p = get_labels(class_,img_nr, 'partial', 1)
                X_p = get_features(img_nr,1)
                scaled = scaler_old.transform(X_p)
                X_new_ = get_features_new(img_nr,1)
                scaled_new = scaler_new.transform(X_new_)
                y_class.append( 1 if y_p[0] > 0 else 0)
                X_all_.append(scaled[0])
                X_new_all.append(scaled_new[0])
            print 'statring'
            # score function: twice iterated 10-fold cross-validated accuracy
            @optunity.cross_validated(x=X_all_, y=y_class, num_folds=10, num_iter=2)
            def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
                model = SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
                decision_values = model.decision_function(x_test)
                return optunity.metrics.roc_auc(y_test, decision_values)
            # perform tuning
            hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])
            svc_old = SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma'])
            svc_new = SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma'])
            print hps['logC'], hps['logGamma']
            exit()
            
            loss__train_mine_old = []
            loss__train_mine_new = []
            loss__train_sgd_old = []
            loss__train_sgd_new = []
            loss__train_mlp_old = []
            loss__train_mlp_new = []
            loss__eval_mine_old = []
            loss__eval_mine_new = []
            loss__eval_sgd_new = []
            loss__eval_sgd_old = []
            loss__eval_mlp_old = []
            loss__eval_mlp_new = []
            loss__train_svc_new = []
            loss__train_svc_old = []
            loss__eval_svc_new = []
            loss__eval_svc_old = []
            epochs_total = 10
            learning_rate = learning_rate0
            
        
            for epochs in np.arange(epochs_total):
                global prune_tree_levels
                prune_tree_levels = 1 
                w_all = {}
                alphas = [1-alpha1,alpha1,reg]
                
                if epochs == 0:
                    # initialize or reset w , plot_losses
                    w_old = 0.01 * np.random.rand(4096)     
                    w_new = 0.01 * np.random.rand(1000)  
                    plot_training_loss = []
                    plot_evaluation_loss = []
                loss_train = []
                loss_test = []
                full_image_test = []
                full_image_train = []
                learning_rate_again = []
                t = 0
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
                loss__train = []
                loss__eval = []
                full_image__train = []
                print epochs, learning_rate, alphas
                X_all = []
                
                X_new = []
                y_all = []
                y_all = []
                y_class = []
                X_all_ = []
                X_new_all = []
                for it,img_nr in enumerate(img_train):
                    y_p = get_labels(class_,img_nr, 'partial', 1)
                    X_p = get_features(img_nr,1)
                    scaled = scaler_old.transform(X_p)
                    X_new_ = get_features_new(img_nr,1)
                    scaled_new = scaler_new.transform(X_new_)
                    if y_p[0]>0:
                        w_old = learn_root(w_old,scaled[0],y_p[0],learning_rate,alphas)
                        w_new = learn_root(w_new,scaled_new[0],y_p[0],learning_rate,alphas)
                        t += 1
                        X_all.append(scaled[0])
                        X_new.append(scaled_new[0])
                        y_all.append(y_p[0])
                    y_class.append( 1 if y_p[0] > 0 else 0)
                    X_all_.append(scaled[0])
                    X_new_all.append(scaled_new[0])
                clf_old.partial_fit(X_all,y_all)
                clf_new.partial_fit(X_new,y_all)
                svc_old.fit(X_all_,y_class)
                svc_new.fit(X_new_all,y_class)
                #TODO: maybe increase tolerance
                for i in range(3):
                    mlp_new.partial_fit(X_new,y_all)
                    mlp_old.partial_fit(X_all,y_all)
                learning_rate = learning_rate0 * (1+learning_rate0*0.01*t)**-1
                
                                    
                print 'compute loss'
                mine_old = []
                mine_new = []
                sgd_old = []
                sgd_new = []
                mlp_new_error = []
                mlp_old_error = []
                svc_new_error = []
                svc_old_error = []
                for img_nr in img_train:
                    y_p = get_labels(class_,img_nr, 'partial', 1)
                    if y_p[0]>0:
                        X_p = get_features(img_nr,1)
                        #print inv[0]
                        scaled = scaler_old.transform(X_p)
                        X_new_ = get_features_new(img_nr,1)
                        scaled_new = scaler_new.transform(X_new_)
                        mine_old.append((max(0,np.dot(w_old, scaled[0])) - y_p[0])**2)
                        mine_new.append((max(0,np.dot(w_new, scaled_new[0])) - y_p[0])**2)
                        y_c = 1 if y_p[0] > 0 else 0

                        svc_new_error.append((svc_new.predict(scaled_new[0].reshape(1, -1)) - y_c) ** 2)
                        svc_old_error.append((svc_old.predict(scaled[0].reshape(1, -1)) - y_c) ** 2)

                        sgd_old.append((clf_old.predict(scaled[0].reshape(1, -1)) - y_p[0])**2)
                        sgd_new.append((clf_new.predict(scaled_new[0].reshape(1, -1)) - y_p[0])**2)
                        mlp_new_error.append((mlp_new.predict(scaled_new[0].reshape(1, -1)) - y_p[0])**2)
                        mlp_old_error.append((mlp_old.predict(scaled[0].reshape(1, -1)) - y_p[0])**2)
                loss__train_mine_new.append(np.array(mine_new).sum() / len(mine_new))
                loss__train_mine_old.append(np.array(mine_old).sum() / len(mine_old))
                loss__train_sgd_new.append(np.array(sgd_new).sum() / len(sgd_new))
                loss__train_sgd_old.append(np.array(sgd_old).sum() / len(sgd_old))
                loss__train_mlp_new.append(np.array(mlp_new_error).sum() / len(mlp_new_error))
                loss__train_mlp_old.append(np.array(mlp_old_error).sum() / len(mlp_old_error))
                loss__train_svc_old.append(np.array(svc_old_error).sum() / len(svc_old_error))
                loss__train_svc_new.append(np.array(svc_new_error).sum() / len(svc_new_error))
                mine_old = []
                mine_new = []
                sgd_old = []
                sgd_new = []
                mlp_new_error = []
                mlp_old_error = []
                svc_new_error = []
                svc_old_error = []
                for img_nr in img_eval:
                    y_p = get_labels(class_,img_nr, 'partial', 1)
                    if y_p[0]>0:
                        X_p = get_features(img_nr,1)
                        X_new_ = get_features_new(img_nr,1)
                        scaled_new = scaler_new.transform(X_new_)
                        
                        mine_old.append((max(0,np.dot(w_old, scaled[0])) - y_p[0])**2)
                        mine_new.append((max(0,np.dot(w_new, scaled_new[0])) - y_p[0])**2)
                        y_c = 1 if y_p[0] > 0 else 0

                        svc_new_error.append((svc_new.predict(scaled_new[0].reshape(1, -1)) - y_c) ** 2)
                        svc_old_error.append((svc_old.predict(scaled[0].reshape(1, -1)) - y_c) ** 2)

                        sgd_old.append((clf_old.predict(scaled[0].reshape(1, -1)) - y_p[0])**2)
                        sgd_new.append((clf_new.predict(scaled_new[0].reshape(1, -1)) - y_p[0])**2)
                        mlp_new_error.append((mlp_new.predict(scaled_new[0].reshape(1, -1)) - y_p[0])**2)
                        mlp_old_error.append((mlp_old.predict(scaled[0].reshape(1, -1)) - y_p[0])**2)
                loss__eval_mine_new.append(np.array(mine_new).sum() / len(mine_new))
                loss__eval_mine_old.append(np.array(mine_old).sum() / len(mine_old))
                loss__eval_sgd_new.append(np.array(sgd_new).sum() / len(sgd_new))
                loss__eval_sgd_old.append(np.array(sgd_old).sum() / len(sgd_old))
                loss__eval_mlp_new.append(np.array(mlp_new_error).sum() / len(mlp_new_error))
                loss__eval_mlp_old.append(np.array(mlp_old_error).sum() / len(mlp_old_error))
                loss__eval_svc_old.append(np.array(svc_old_error).sum() / len(svc_old_error))
                loss__eval_svc_new.append(np.array(svc_new_error).sum() / len(svc_new_error))
                            
            mse_classes['%s_%s'%(class_,learning_rate0)] = [loss__train_mine_new[-1],loss__train_mine_old[-1], loss__train_sgd_new[-1], loss__train_sgd_old[-1],loss__train_mlp_new[-1],loss__train_mlp_old[-1],loss__train_svc_old[-1],loss__train_svc_new[-1],loss__eval_mine_new[-1],loss__eval_mine_old[-1],loss__eval_sgd_new[-1],loss__eval_sgd_old[-1],loss__eval_mlp_new[-1],loss__eval_mlp_old[-1],loss__eval_svc_old[-1],loss__eval_svc_new[-1]]
    with open('/home/tstahl/mse_classes_clipped.pickle', 'wb') as handle:
        pickle.dump(mse_classes, handle)
    for cc in mse_classes:
        print cc, np.round(mse_classes[cc],2)


def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3]), 1/((A[2]- A[0] + 1)*(A[3]-A[1] + 1))
        #return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom ||r2.bottom < r1.top);
    
def learn_root(w,x,y,learning_rate,alphas):
    inner_prod = 0.0
    for f_ in range(len(w)):
        inner_prod += (w[f_] * x[f_])
    # clip negative predictions
    #dloss = inner_prod - y
    dloss = max(0,inner_prod) - y
    for f_ in range(len(w)):
        w[f_] += (learning_rate * ((-x[f_] * dloss)))# + alphas[1] * w[f_]))
    return w
    
if __name__ == "__main__":
#    cProfile.run('main()')
    main()
