import IEP
import Input
import numpy as np
import Data
from sklearn.preprocessing import MinMaxScaler
import time



class SGD:
    def __init__(self, mode, category, prune_tree_levels, batch_size, eta, gamma, alpha, num_features=1000):
        self.prune_tree_levels = prune_tree_levels
        self.n_features = num_features
        self.w = 0.0001 * np.random.rand(self.n_features)
        self.predictor = IEP.IEP(self.w, 'prediction')
        self.w_update = np.zeros(self.n_features)
        self.learner = IEP.IEP(1, 'learning')
        self.load = Input.Input('pascal', category)
        self.batch_size = batch_size
        self.samples_seen = 0
        self.eta = eta
        self.eta0 = eta
        self.gamma = gamma
        self.alpha = alpha
        self.functions = {}
        if mode == 'max':
            self.method = self.learn_max
            self.loss = self.loss_max
            self.predict = self.predict_max
        elif mode == 'avg':
            self.method = self.learn_mean
            self.loss = self.loss_mean
            self.predict = self.predict_mean
            
    def set_scaler(self, scaler):
        self.scaler = scaler
        
    def loss_max(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return (np.max(level_preds) - img_data.y)**2 + self.alpha * np.dot(self.w,self.w)
        
    def loss_mean(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return (np.mean(level_preds) - img_data.y)**2 + self.alpha * np.dot(self.w,self.w)
        
    def predict_mean(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.mean(level_preds)
        
    def predict_max(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, {})
        return np.max(level_preds)
        
        
    def evaluate(self, mode, to=-1):
        squared_error = 0.0
        error = 0.0
        non_zero_error = 0.0
        n_non_zero = 0.0
        if mode == 'train':
            numbers = self.load.training_numbers
        elif mode == 'test':
            numbers = self.load.test_numbers
        for img_nr in numbers[:to]:
            img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            img_data.scale(self.scaler)
            img_loss = self.predict(img_data)
            squared_error += img_loss ** 2
            error += abs(img_loss)
            if img_data.y > 0:
                non_zero_error += img_loss ** 2
                n_non_zero += 1
        return squared_error/len(numbers), error / len(numbers), non_zero_error / n_non_zero
        
        
        
    def learn(self, to=-1):
        training_data = self.load.training_numbers
        for i_img_nr, img_nr in enumerate(training_data[:to]):
            img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            img_data.scale(self.scaler)
            if img_nr in self.functions:
                img_functions = self.functions[img_nr]
                self.w_update += self.method(img_data, img_functions)[0]
            else:
                temp = {}
                upd, fct = self.method(img_data, temp)
                self.w_update += upd
                self.functions[img_nr] = fct
            self.samples_seen += 1
            
            if i_img_nr%self.batch_size == 0:
                self.update()
        self.update()
        self.predictor = IEP.IEP(self.w, 'prediction')
        
        
    def update(self):
        #print 'updating'
        self.w -= self.eta * self.w_update
        self.eta = self.eta * (1+self.eta0*self.gamma*self.samples_seen)**-1
        self.w_update = np.zeros(self.n_features)
        self.predictor = IEP.IEP(self.w, 'prediction')
        
        
    def learn_max(self, img_data, functions):
        level_preds, functions = self.predictor.get_iep_levels(img_data, functions)
        print 'preds: ',img_data.img_nr, level_preds, ' y: ', img_data.y
        
        #print level_preds
        ind_max = level_preds.index(max(level_preds))
        upd, _ = self.learner.iep(img_data, [], ind_max)#functions[ind_max]
        return (self.predict(img_data) - img_data.y) * upd + self.alpha * self.w, functions
        
    def learn_mean(self, img_data, functions):
        iep_levels, functions = self.learner.get_iep_levels(img_data, functions)
        return (self.predict(img_data) - img_data.y) * (np.array(iep_levels).sum() / len(iep_levels)) + self.alpha * self.w, functions