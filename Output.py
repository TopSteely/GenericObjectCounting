import matplotlib
matplotlib.use('agg')
import pickle
import matplotlib.pyplot as plt
from scipy.misc import imread
from matplotlib.patches import Rectangle
import numpy as np
import pylab as pl


class Output:
    def __init__(self, mode, category, prune_tree_levels, experiment):
        self.category = category
        self.prune_tree_levels = prune_tree_levels
        self.mode = mode
        self.experiment = experiment
        self.mse_path = "/home/tstahl/plot/%s_%s_mse_%s_%s.p"
        self.ae_path = "/home/tstahl/plot/%s_%s_ae_%s_%s.p"
        self.nn_path = "/home/tstahl/plot/%s_%s_nn_%s_%s.p"
        self.npe_path = "/home/tstahl/plot/%s_%s_npe_%s_%s.p"
        self.model_path = "/var/node436/local/tstahl/models/%s_%s_%s_%s.p"
        self.plot_path = "/var/node436/local/tstahl/plos/%s_%s.png"
        self.image_path = "/var/node436/local/tstahl/Images/%s.jpg"
        
        
        
    def save(self, mse_level, ae_level, nn, sgd):
        pickle.dump(mse_level, open( self.mse_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(ae_level, open( self.ae_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(nn, open( self.nn_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(sgd.w, open( self.model_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        #pickle.dump(num_per_image, open( self.npe_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        
    def plot_preds(self, preds, preds_skl, y, alpha):
        sorted_preds = []
        sorted_preds_skl = []
        sorted_y = []
        decorated = [(y_i, i) for i, y_i in enumerate(y)]
        decorated.sort()
        for y_i, i in reversed(decorated):
            sorted_preds.append(preds[i])
            sorted_preds_skl.append(preds_skl[i])
            sorted_y.append(y_i)
        plt.figure()
        plt.plot(range(len(preds)), preds, 'ro',label='prediction')
        plt.plot(range(len(preds)), preds_skl, 'gD', label='sklearn')
        plt.plot(range(len(preds)), y, 'y*',label='target')
        plt.ylabel('y')
        plt.ylim([-1,len(preds)+1])
        plt.xlim([-1,len(preds)+1])
        plt.legend(loc='upper center')
        plt.title('%s'%(alpha))
        plt.savefig(self.plot_path%(self.mode,alpha))     
        
    def plot_level_boxes(self, rects, img_nr):
        colors = ['red','blue','green']
        im = imread(self.image_path%(format(img_nr, "06d")))
        plt.imshow(im)
        plt.axis('off')
        ax = plt.gca()
        for rect,c in zip(rects,colors):
        #ax.add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), edgecolor='black', facecolor='none'))
            ax.add_patch(Rectangle((int(rect[0]), int(rect[1])), int(rect[2] - rect[0]), int(rect[3] - rect[1]), edgecolor=c, facecolor=c, alpha=0.5))
        
    
        plt.savefig(self.plot_path%(img_nr,'rects'))
        plt.clf()
        f,ax = plt.subplots(3)
        ax[0].imshow(im[rects[0][1]:rects[0][3], rects[0][0]:rects[0][2]])
        ax[1].imshow(im[rects[1][1]:rects[1][3], rects[1][0]:rects[1][2]])
        ax[2].imshow(im[rects[2][1]:rects[2][3], rects[2][0]:rects[2][2]])
        plt.savefig(self.plot_path%(img_nr,'sub_rects'))
        
        