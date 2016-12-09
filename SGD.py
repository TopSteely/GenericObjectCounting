import IEP
import Input
import numpy as np
import Data
from sklearn.preprocessing import MinMaxScaler



class SGD:
    def __init__(self, mode, categoy, prune_tree_levels, batch_size, eta, gamma, alpha):
        self.prune_tree_levels = prune_tree_levels
        self.n_features = 1000
        self.w = 0.0001 * np.random.rand(self.n_features)
        self.w_update = np.zeros(self.n_features)
        self.learner = IEP.IEP(1, 'learning')
        self.predictor = None
        self.Input = Input.Input('pascal', categoy)
        self.batch_size = batch_size
        self.samples_seen = 0
        self.eta = eta
        self.eta0 = eta
        self.gamma = gamma
        self.alpha = alpha
        self.scaler = MinMaxScaler()
        self.functions = []
        if mode == 'max':
            self.method = self.learn_max
            self.loss = self.loss_max
        elif mode == 'avg':
            self.method = self.learn_mean
            self.loss = self.loss_mean
        
    def loss_max(self, img_data, functions):
        level_preds = self.predictor.get_iep_levels(img_data, functions)
        return (np.max(level_preds) - img_data.y)**2 + self.alpha * np.dot(self.w,self.w)
        
    def loss_mean(self, img_data, functions):
        level_preds = self.predictor.get_iep_levels(img_data, functions)
        return (np.mean(level_preds) - img_data.y)**2 + self.alpha * np.dot(self.w,self.w)
        
        
    def learn(self):
        training_data = self.Input.training_numbers
        for i_img_nr, img_nr in enumerate(training_data):
            img_data = Data.Data(self.Input, img_nr)
            if img_nr in self.functions:
                img_functions = self.functions[img_nr]
                self.w_update += self.method(img_data, img_functions)
            else:
                upd, fct = self.method(img_data, [])
                self.w_update += upd
                self.functions[img_nr] = fct
            
            
            if i_img_nr%self.batch_size == 0:
                self.update()
        self.update()
        self.predictor = IEP.IEP(self.w, 'prediction')
        
        
    def update(self):
        self.w -= self.eta * self.w_update
        self.eta = self.eta * (1+self.eta0*self.gamma*self.samples_seen)**-1
        self.w_update = np.zeros(self.n_features)
        
        
    def learn_max(self, img_data, functions):
        level_preds = self.predictor.get_iep_levels(img_data, functions)
        ind_max = level_preds.index(max(level_preds))
        return self.learner.iep(img_data, functions[ind_max], ind_max) + self.alpha * self.w, functions
        
    def learn_mean(self, img_data, functions):
        iep_levels = self.learner.get_iep_levels(img_data, functions)
        return np.array(iep_levels).sum() / len(iep_levels) + self.alpha * self.w, functions