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
        self.mse_path = "/home/tstahl/plot/%s_%s_mse_%s_%s_%s_%s_%s.p"
        self.ae_path = "/home/tstahl/plot/%s_%s_ae_%s_%s_%s_%s_%s.p"
        self.nn_path = "/home/tstahl/plot/%s_%s_nn_%s_%s_%s_%s_%s.p"
        self.npe_path = "/home/tstahl/plot/%s_%s_npe_%s_%s.p"
        self.model_path = "/var/node436/local/tstahl/models/%s_%s_%s_%s_%s_%s_%s.p"
        self.plot_path = "/var/node436/local/tstahl/plos/%s_%s_%s.png"
        self.preds_plot_path = "/var/node436/local/tstahl/plos/preds_%s_%s_%s_%s.png"
        self.image_path = "/var/node436/local/tstahl/Images/%s.jpg"
        self.scaler_path = '/var/node436/local/tstahl/models/scaler_dennis.p'
        if mode.startswith('dennis'):
                self.scaler_category_path = '/var/node436/local/tstahl/models/scaler_%s_dennis.p'%(category)
        elif mode.startswith('pascal'):
            self.scaler_category_path = '/var/node436/local/tstahl/models/scaler_%s_pascal.p'%(category)
            self.classifier_path = '/var/node436/local/tstahl/models/classifier_%s.p'%(category)
        self.feat_var_path = '/var/node436/local/tstahl/plos/feat_var.png'
        self.loss_path = '/var/node436/local/tstahl/plos/loss_%s_%s_%s_%s_%s_%s.png'
        self.best_path = '/var/node436/local/tstahl/plos/best_%s_%s_%s.png'
        self.upd_path = '/var/node436/local/tstahl/plos/upd_%s.png'
        
    def dump_scaler(self, scaler):
        pickle.dump(scaler, open(self.scaler_path, "wb"))
        
    def dump_scaler_category(self, scaler):
        pickle.dump(scaler, open(self.scaler_category_path, "wb"))
        
        
    def dump_classifier(self, classifier):
        pickle.dump(classifier, open(self.classifier_path, "wb"))
        
        
    def save(self, mse_level, ae_level, nn, sgd, eta0, alpha, learn_mode):
        pickle.dump(mse_level, open( self.mse_path%(self.experiment, self.mode, self.category, self.prune_tree_levels, eta0, alpha, learn_mode), "wb" ))
        pickle.dump(ae_level, open( self.ae_path%(self.experiment, self.mode, self.category, self.prune_tree_levels, eta0, alpha, learn_mode), "wb" ))
        pickle.dump(nn, open( self.nn_path%(self.experiment, self.mode, self.category, self.prune_tree_levels, eta0, alpha, learn_mode), "wb" ))
        pickle.dump(sgd.w, open( self.model_path%(self.experiment, self.mode, self.category, self.prune_tree_levels, eta0, alpha, learn_mode), "wb" ))
        #pickle.dump(num_per_image, open( self.npe_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        
    def plot_preds(self, preds, y, alpha, dataset):
        if self.mode.endswith('multi'):
            f,ax = plt.subplots(self.prune_tree_levels)
            for lvl in range(self.prune_tree_levels):
                ax[lvl].plot(preds[lvl], 'ro', label="prediction")
                ax[lvl].plot(y, 'y*', label="target")
                ax[lvl].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off')
                ax[lvl].title.set_text("Prediction level %s"%(lvl))
            #plt.legend('upper left')
        else:
            sorted_preds = []
            sorted_y = []
            decorated = [(y_i, i) for i, y_i in enumerate(y)]
            decorated.sort()
            for y_i, i in reversed(decorated):
                sorted_preds.append(preds[i])
                sorted_y.append(y_i)
            plt.figure()
            plt.plot(range(len(preds)), preds, 'ro',label='prediction')
            plt.plot(range(len(preds)), y, 'y*',label='target')
            plt.ylabel('y')
            plt.ylim([-1,len(preds)+1])
            plt.xlim([-1,len(preds)+1])
            plt.legend(loc='upper center')
            plt.title('%s'%(alpha))
        plt.savefig(self.preds_plot_path%(self.mode,alpha,self.category, dataset))     
        
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

    def plot_features_variance(self, var1, var2):
        f,ax = plt.subplots(2)
        ax[0].plot(var1, "rx")
        ax[1].plot(var2, "bo")
        plt.savefig(self.feat_var_path)

    def plot_train_val_loss_old(self, train, val, eta, alpha):
        plt.clf()
        plt.plot(train, '-rx', label="training")
        plt.plot(val, '-bo', label="validation")
        plt.legend()
        plt.savefig(self.loss_path%(self.experiment,self.prune_tree_levels,eta,self.category, alpha))
        

    def plot_train_val_loss(self, train, val, eta, alpha):
        plt.clf()
        f,ax = plt.subplots(self.prune_tree_levels+1)
        for lvl in range(self.prune_tree_levels+1):
            ax[lvl].plot(train[lvl], '-rx', label="training")
            ax[lvl].plot(val[lvl], '-bo', label="validation")
            ax[lvl].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            if lvl == self.prune_tree_levels:
                ax[lvl].title.set_text("Mean and final loss")
            else:
                ax[lvl].title.set_text("Loss for level %s"%(lvl))
        plt.legend('upper left')
        plt.savefig(self.loss_path%(self.experiment,self.prune_tree_levels,eta,self.category, alpha, self.mode))
        

    def plot_best(self, level_preds, max_level_window):
        #max_level_window = [img_data.boxes[ind],max_level_pred]
        for img_nr in level_preds.keys():
            # in case i want only imgs with levels higher than
            #if len(level_preds[i_img]) < 6:
            #        continue
            im = imread('/var/node436/local/tstahl/Images/'+ (format(img_nr, "06d")) +'.jpg')
            for lvl,(lvl_pred,b_patch) in enumerate(zip(level_preds[img_nr], max_level_window[img_nr])):
                coord_iep = b_patch[0]
                plt.imshow(im)
                plt.axis('off')
                ax = plt.gca()
                #ax.add_patch(Rectangle((int(coord[0]), int(coord[1])), int(coord[2] - coord[0]), int(coord[3] - coord[1]), edgecolor='black', facecolor='none'))
                ax.add_patch(Rectangle((int(coord_iep[0]), int(coord_iep[1])), int(coord_iep[2] - coord_iep[0]), int(coord_iep[3] - coord_iep[1]), edgecolor='red', facecolor='none'))
                ax.set_title('best Patch: %s\n IEP Level: %s'%(b_patch[1],lvl_pred))
                
            
                plt.savefig(self.best_path%(self.category,img_nr,lvl))
                plt.clf()

    def plot_updates(self,updates1, updates2, updates3):
        plt.plot([upd[0] for upd in updates1], '-ro', label='old')
        plt.plot([upd[1] for upd in updates1], '-rx', label='norm old')
        plt.plot([upd[0] for upd in updates2], '-go', label='new')
        plt.plot([upd[1] for upd in updates2], '-gx', label='norm new')
        plt.plot([upd[0] for upd in updates3], '-bo', label='dummy')
        plt.plot([upd[1] for upd in updates3], '-bx', label='norm dummy')
        #plt.legend()
        plt.savefig(self.upd_path%(self.category))