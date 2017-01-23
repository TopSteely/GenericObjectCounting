import IEP
import Input
import numpy as np
import Data
import random
import time
#from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
import math



class SGD:
    def __init__(self, dataset, mode, category, prune_tree_levels, batch_size, eta, gamma, alpha, num_features=1000):
        print 'init SGD'
        self.prune_tree_levels = prune_tree_levels
        self.n_features = num_features
        self.w = np.zeros(self.n_features)#0.1 * np.random.rand(self.n_features)
        self.predictor = IEP.IEP(self.w, 'prediction')
        self.w_update = np.zeros(self.n_features)
        self.learner = IEP.IEP(1, 'learning')
        self.load = Input.Input(dataset, category)
        self.batch_size = batch_size
        self.samples_seen = 0
        self.eta = eta
        self.eta0 = eta
        self.gamma = gamma
        self.alpha = alpha
        self.functions = {}
        self.sgd = SGDRegressor(eta0=eta, learning_rate='invscaling', shuffle=True, average=True, alpha=alpha)
        if mode == 'max':
            self.method = self.learn_max
            self.loss = self.loss_max
            self.predict = self.predict_max
        elif mode == 'mean':
            self.method = self.learn_mean
            self.loss = self.loss_mean
            self.predict = self.predict_mean
        elif mode == 'meanmax':
            self.method = self.learn_mean
            self.loss = self.loss_mean
            self.predict = self.predict_max
            
    def set_scaler(self, scaler):
        self.scaler = scaler
        
    def loss_max(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return (np.max(level_preds) - img_data.y)**2 + self.alpha * math.sqrt(np.dot(self.w,self.w))
        
    def loss_mean(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return (np.mean(level_preds) - img_data.y)**2 + self.alpha * math.sqrt(np.dot(self.w,self.w))
        
    def predict_mean(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.mean(level_preds)
        
    def predict_max(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.max(level_preds)
        
        
    def evaluate(self, mode, to=-1, debug=False):
        squared_error = 0.0
        error = 0.0
        non_zero_error = 0.0
        n_non_zero = 0.0
        skl_error = 0.0
        if debug:
            preds_d = []
            y_d = []
            preds_skl = []
        if mode == 'train':
            numbers = self.load.category_train[:to]
        elif mode == 'test':
            numbers = self.load.test_numbers[:to]
        elif mode == 'val_cat':
            numbers = self.load.category_val[:to]
        elif mode == 'val_all':
            numbers = self.load.val_numbers[:to]

        for img_nr in numbers:
            img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            img_loss = (self.predict(img_data) - img_data.y)**2
	    #print 'preds: ',img_data.img_nr, self.predict(img_data), ' y: ', img_data.y
            #print 'preds: ',img_data.img_nr, self.predict(img_data), ' y: ', img_data.y, ' sklearn: ', self.sgd.predict(img_data.X[img_data.levels[0][0]].reshape(1, -1))
            squared_error += img_loss
            error += abs(self.predict(img_data) - img_data.y)
            if img_data.y > 0:
                non_zero_error += img_loss
                n_non_zero += 1
            if debug:
                skl_error += (self.sgd.predict(img_data.X[img_data.levels[0][0]].reshape(1, -1)) - img_data.y)**2
                preds_d.append(self.predict(img_data))
                y_d.append(img_data.y)
                preds_skl.append(self.sgd.predict(img_data.X[img_data.levels[0][0]].reshape(1, -1)))
        if debug:
            return preds_d, preds_skl, y_d
        return squared_error/len(numbers), error / len(numbers), non_zero_error / n_non_zero#skl_error/len(numbers),, self.eta
        
        
    def loss_all(self, to=-1):
	tra_loss_temp = 0.0
	te_loss_temp = 0.0
	for img_nr in self.load.category_train[0:to]:
            img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
	    tra_loss_temp += self.loss(img_data)
	for img_nr in self.load.category_val[0:to]:
            img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
	    te_loss_temp += self.loss(img_data)
	return tra_loss_temp/len(self.load.category_train), te_loss_temp/len(self.load.category_val)
        
    def learn(self, instances='all', to=-1, debug=False):
        train_losses = []
        test_losses = []
        if instances=='all':
            training_data = self.load.training_numbers
        else:
            training_data = self.load.category_train
        subset = training_data[:to]
        random.shuffle(subset)
        for i_img_nr, img_nr in enumerate(subset):
            start = time.time()
            img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            if img_nr in self.functions:
                img_functions = self.functions[img_nr]
                upd,_ = self.method(img_data, img_functions)
                self.w_update += upd
            else:
                temp = {}
                upd, fct = self.method(img_data, temp)
                self.w_update += upd
                #self.functions[img_nr] = fct
            self.samples_seen += 1
            if self.prune_tree_levels == 1:
                to_fit = img_data.X[img_data.levels[0][0]].reshape(1, -1)
                self.sgd.partial_fit(to_fit,[img_data.y])
            if (i_img_nr + 1)%self.batch_size == 0:
                self.update()
                if debug:
                    tr_loss, te_loss = self.loss_all(to)
                    train_losses.append(tr_loss)
                    test_losses.append(te_loss)
        if (i_img_nr + 1)%self.batch_size != 0:
            self.update()
            if debug:
                tr_loss, te_loss = self.loss_all()
                train_losses.append(tr_loss)
                test_losses.append(te_loss)
        if debug:
    	   return train_losses, test_losses
        
        
    def update(self):
        self.w -= (self.eta * self.w_update)
        self.eta = self.eta * (1+self.eta0*self.gamma*self.samples_seen)**-1
        self.w_update = np.zeros(self.n_features)
        self.predictor = IEP.IEP(self.w, 'prediction')
        
        
    def learn_max(self, img_data, functions):
        level_preds, functions = self.predictor.get_iep_levels(img_data, functions)
        #print 'preds: ',img_data.img_nr, level_preds, ' y: ', img_data.y
        
        #print level_preds
        ind_max = level_preds.index(max(level_preds))
        upd, _ = self.learner.iep(img_data, [], ind_max)#functions[ind_max]
        return 2 * (self.predict(img_data) - img_data.y) * upd + 2 * self.alpha * self.w, functions
        
    def learn_mean(self, img_data, functions):
        iep_levels, functions = self.learner.get_iep_levels(img_data, functions)
        #temp1 = 2 * (self.predict(img_data) - img_data.y) * (np.sum(iep_levels,axis=0) / len(iep_levels))
        #temp2 = 2 * self.alpha * self.w
        return 2 * (self.predict(img_data) - img_data.y) * (np.sum(iep_levels,axis=0) / len(iep_levels)) + 2 * self.alpha * self.w, functions
