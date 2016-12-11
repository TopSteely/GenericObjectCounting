import pickle

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
        
        
        
    def save(self, mse_level, ae_level, nn):
        pickle.dump(mse_level, open( self.mse_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(ae_level, open( self.ae_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        pickle.dump(nn, open( self.nn_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))
        #pickle.dump(num_per_image, open( self.npe_path%(self.experiment, self.mode, self.category, self.prune_tree_levels), "wb" ))