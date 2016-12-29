import matplotlib
matplotlib.use('agg')
import pickle
import matplotlib.pyplot as plt

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
        self.plot_path = "/var/node436/local/tstahl/plos/%s.png"
        
        
        
    def save(self, mse_level, ae_level, nn, sgd):
        pickle.dump(mse_level, open( self.mse_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(ae_level, open( self.ae_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(nn, open( self.nn_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(sgd.w, open( self.model_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        #pickle.dump(num_per_image, open( self.npe_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        
    def plot_preds(self, preds, preds_skl, y):
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
        plt.plot(range(len(preds)), preds, 'r*',label='prediction')
        plt.plot(range(len(preds)), preds_skl, 'gD', label='sklearn')
        plt.plot(range(len(preds)), y, 'yo',label='target')
        plt.ylabel('y')
        plt.ylim([-1,2])
        plt.xlim([-1,7])
        plt.legend(loc='upper center')
        plt.savefig(self.plot_path%(self.mode))        
        