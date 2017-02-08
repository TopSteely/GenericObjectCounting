import IEP
import Input
import numpy as np
import Data
import BlobData
import random
import time
#from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
import math



class SGD:
    def __init__(self, dataset, mode, category, prune_tree_levels, batch_size, eta, gamma, alpha, num_features=1000):
        print 'init SGD'
        self.version = mode
        self.prune_tree_levels = prune_tree_levels
        self.n_features = num_features
        self.w = np.zeros(self.n_features)#0.1 * np.random.rand(self.n_features)
        self.predictor = IEP.IEP(self.w, 'prediction')
        self.w_update = np.zeros((self.prune_tree_levels,self.n_features))
        self.learner = IEP.IEP(1, 'learning')
        self.dataset = dataset
        self.load = Input.Input(dataset, category, prune_tree_levels)
        self.batch_size = batch_size
        self.samples_seen = 0
        self.eta = eta
        self.eta0 = eta
        self.gamma = gamma
        self.alpha = alpha
        self.functions = {}
        self.scaler = None
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
        elif mode == 'old':
            self.method = self.learn_old
            #self.loss = self.loss_mean
            self.predict = self.predict_max
        elif mode == 'ind':
            self.method = self.learn_ind
            #self.loss = self.loss_mean
            self.predict = self.predict_max
        elif mode == 'multi':
            self.method = self.learn_multi
            self.loss = self.loss_multi
            self.predict = self.predict_multi
            self.w_multi = np.zeros((self.prune_tree_levels,self.n_features))
            self.w_update = np.zeros((self.prune_tree_levels,self.n_features))
        elif mode == 'new':
            self.method = self.learn_new
            self.loss = self.loss_new
            self.predict = self.predict_new
        #blob dataset, have to save the data because of random bbox creation
        if dataset == 'blob':
            self.blobtraindata = []
            self.blobtestdata = []
            for img_nr in self.load.training_numbers:
                tmp_data = BlobData.BlobData(self.load, img_nr, self.scaler, prune_tree_levels)
                self.blobtraindata.append(tmp_data)
            for img_nr in self.load.val_numbers:
                tmp_data = BlobData.BlobData(self.load, img_nr, self.scaler, prune_tree_levels)
                self.blobtestdata.append(tmp_data)

    def reset_w(self):
        self.w_update = np.zeros((self.prune_tree_levels,self.n_features))
        self.w = np.zeros(self.n_features)
        self.predictor = IEP.IEP(self.w, 'prediction')
        if self.version == 'multi':
            self.w_multi = np.zeros((self.prune_tree_levels,self.n_features))
            self.w_update = np.zeros((self.prune_tree_levels,self.n_features))

    def set_scaler(self, scaler):
        self.scaler = scaler
        
    def loss_max(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.min(np.array(level_preds) - img_data.y)**2 + self.alpha * math.sqrt(np.dot(self.w,self.w))

    def loss_new(self, img_data):
        #for 
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.min(np.array(level_preds) - img_data.y)**2 + self.alpha * math.sqrt(np.dot(self.w,self.w))
        
    def loss_mean(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.mean(np.array(level_preds) - img_data.y)**2 + self.alpha * math.sqrt(np.dot(self.w,self.w))

    def loss_multi(self, img_data):
        level_preds = []
        for lvl in range(self.prune_tree_levels):
            if lvl >= len(img_data.levels):
                level_pred = level_preds[-1]
            else:
                predictor = IEP.IEP(self.w_multi[lvl], 'prediction')
                level_pred, _ = predictor.iep(img_data, [], lvl)
            level_preds.append(level_pred)
        return np.mean(np.array(level_preds) - img_data.y)**2 + self.alpha * math.sqrt(np.dot(self.w,self.w))

    def loss_per_level(self, img_data):
        if self.version == 'multi':
            level_preds = []
            for lvl in range(self.prune_tree_levels):
                if lvl >= len(img_data.levels):
                    level_pred = level_preds[-1]
                else:
                    predictor = IEP.IEP(self.w_multi[lvl], 'prediction')
                    level_pred, _ = predictor.iep(img_data, [], lvl)
                level_preds.append(level_pred)
        else:
            level_preds, _ = self.predictor.get_iep_levels(img_data, [])
            # for images with less levels than prune_tree_levels, just append the last level
            if len(level_preds) < self.prune_tree_levels:
                for missing in range(self.prune_tree_levels - len(level_preds)):
                    level_preds.append(level_preds[-1])
        return (np.array(level_preds) - img_data.y)**2 + self.alpha * math.sqrt(np.dot(self.w,self.w))
        
    def predict_mean(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.mean(level_preds)
        
    def predict_max(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return np.max(level_preds)

    def predict_old(self, img_data, level):
        level_pred, _ = self.predictor.iep(img_data, [], level)
        return level_pred

    def predict_ind(self, img_data):
        level_preds, _ = self.predictor.get_iep_levels(img_data, [])
        return level_preds

    def predict_multi(self, img_data):
        preds = []
        for level in range(self.prune_tree_levels):
            if level >= len(img_data.levels):
                print 'cannot evaluate, using last possible level'
                level_pred = preds[-1]
            else:
                predictor = IEP.IEP(self.w_multi[level], 'prediction')
                level_pred, _ = predictor.iep(img_data, [], level)
            preds.append(level_pred)
        return np.array(preds) #np.mean(preds)
        
        
    def evaluate(self, mode, to=-1, debug=False):
        if self.version == 'multi':
            squared_error = np.zeros(self.prune_tree_levels)
            error = np.zeros(self.prune_tree_levels)
            non_zero_error = np.zeros(self.prune_tree_levels)
            n_non_zero = np.zeros(self.prune_tree_levels)
            skl_error = np.zeros(self.prune_tree_levels)
        else:
            squared_error = 0.0
            error = 0.0
            non_zero_error = 0.0
            n_non_zero = 0.0
            skl_error = 0.0
        if debug:
            if self.version == 'multi':
                preds_d = np.array([], dtype=np.int64).reshape(self.prune_tree_levels,0)
            else:
                preds_d = []
            y_d = []
            preds_skl = []
        if mode == 'train_cat':
            numbers = self.load.category_train[:to]
        if mode == 'train_all':
            numbers = self.load.training_numbers[:to]
        elif mode == 'test':
            numbers = self.load.test_numbers[:to]
        elif mode == 'val_cat':
            numbers = self.load.category_val[:to]
        elif mode == 'val_all':
            numbers = self.load.val_numbers[:to]
        elif mode == 'blobtrain':
            numbers = self.load.training_numbers[:to]
            b_data = self.blobtraindata
        elif mode == 'blobtest':
            numbers = self.load.val_numbers
            b_data = self.blobtestdata
        elif mode == 'val_category_levels':
            numbers = self.load.category_val_with_levels[:to]
        elif mode == 'train_category_levels':
            numbers = self.load.category_train_with_levels[:to]

        print 'evaluate Len: ', numbers

        for i_img_nr,img_nr in enumerate(numbers):
            if self.dataset == 'blob':
                img_data = b_data[i_img_nr]
            else:
                if self.n_features == 1:
                    img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features, True)
                else:
                    img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            print self.predict(img_data), img_data.y
            img_loss = (self.predict(img_data) - img_data.y) ** 2
	    #print 'preds: ',img_data.img_nr, self.predict(img_data), ' y: ', img_data.y
            #print 'preds: ',img_data.img_nr, self.predict(img_data), ' y: ', img_data.y, ' sklearn: ', self.sgd.predict(img_data.X[img_data.levels[0][0]].reshape(1, -1))
            squared_error += img_loss
            error += abs(self.predict(img_data) - img_data.y)
            if img_data.y > 0:
                non_zero_error += img_loss
                n_non_zero += 1
            if debug:
                #skl_error += (self.sgd.predict(img_data.X[img_data.levels[0][0]].reshape(1, -1)) - img_data.y)**2
                if self.version == 'multi':
                    tmptmp = self.predict(img_data)
                    preds_d = np.concatenate((preds_d,self.predict(img_data).reshape(-1,1)), axis=1)
                else:
                    preds_d.append(self.predict(img_data))
                y_d.append(img_data.y)
                #preds_skl.append(self.sgd.predict(img_data.X[img_data.levels[0][0]].reshape(1, -1)))
        if debug:
            return preds_d, y_d#,  preds_skl
        return squared_error/len(numbers), error / len(numbers), non_zero_error / n_non_zero#skl_error/len(numbers),, self.eta
        
        
    def loss_all(self, to=-1):
        tra_loss_temp = 0.0
        te_loss_temp = 0.0
        for img_nr in self.load.category_train[0:to]:
            if self.n_features == 1:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features, True)
            else:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            tra_loss_temp += self.loss(img_data)
        for img_nr in self.load.category_val[0:to]:
            if self.n_features == 1:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features, True)
            else:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            te_loss_temp += self.loss(img_data)
        return tra_loss_temp/len(self.load.category_train), te_loss_temp/len(self.load.category_val)

    def loss_per_level_all(self, instances, to=-1):
        tra_loss_temp = np.zeros(self.prune_tree_levels+1)
        te_loss_temp = np.zeros(self.prune_tree_levels+1)
        if instances == 'all':
            training_ims = self.training_numbers[0:to]
            validation_ims = self.val_numbers[0:to]
        elif instances == 'category':
            training_ims = self.load.category_train[0:to]
            validation_ims = self.load.category_val[0:to]
        elif instances == 'category_levels':
            training_ims = self.load.category_train_with_levels[0:to]
            validation_ims = self.load.category_val_with_levels[0:to]
        for img_nr in training_ims:
            if self.n_features == 1:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features, True)
            else:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            #print self.loss_per_level(img_data).shape, tra_loss_temp[0:self.prune_tree_levels].shape, self.loss_per_level(img_data).reshape(1,-1).shape
            tra_loss_temp[0:self.prune_tree_levels] += self.loss_per_level(img_data).reshape(self.prune_tree_levels,)
            tra_loss_temp[self.prune_tree_levels] += self.loss(img_data)
        for img_nr in validation_ims:
            if self.n_features == 1:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features, True)
            else:
                img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
            te_loss_temp[0:self.prune_tree_levels] += self.loss_per_level(img_data)
            te_loss_temp[self.prune_tree_levels] += self.loss(img_data)
        return tra_loss_temp/len(self.load.category_train), te_loss_temp/len(self.load.category_val)
        
    def learn(self, instances='all', to=-1, debug=False):
        train_losses = np.array([], dtype=np.int64).reshape(self.prune_tree_levels+1,0)
        test_losses = np.array([], dtype=np.int64).reshape(self.prune_tree_levels+1,0)
        if instances=='all':
            training_data = self.load.training_numbers
        elif instances=='category':
            training_data = self.load.category_train
        elif instances=='category_levels':
            training_data = self.load.category_train_with_levels
        subset = training_data[:to]
        print 'Len: ', len(training_data)
        random.shuffle(subset)
        for i_img_nr, img_nr in enumerate(subset):
            print 'img', img_nr
            start = time.time()
            if self.dataset == 'blob':
                img_data = self.blobtraindata[i_img_nr]
            else:
                if self.n_features == 1:
                    img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features, True)
                else:
                    img_data = Data.Data(self.load, img_nr, self.prune_tree_levels, self.scaler, self.n_features)
                if instances=='category_levels':
                    assert len(img_data.levels) >= self.prune_tree_levels
            if img_nr in self.functions:
                img_functions = self.functions[img_nr]
                print 'getting fct ', img_functions
                upd,_ = self.method(img_data, img_functions)
                self.w_update += upd
            else:
                temp = []
                if self.version == 'old':
                    self.method(img_data, temp)
                else:
                    upd, fct = self.method(img_data, temp)
                    print 'creating', fct
                    self.w_update += upd
                    self.functions[img_nr] = fct
            self.samples_seen += 1
            if self.prune_tree_levels == 1:
                to_fit = img_data.X[img_data.levels[0][0]].reshape(1, -1)
                self.sgd.partial_fit(to_fit,[img_data.y])
            if (i_img_nr + 1)%self.batch_size == 0:
                self.update()
                if debug:
                    tr_loss, te_loss = self.loss_per_level_all(instances, to)
                    train_losses = np.concatenate((train_losses,tr_loss.reshape(-1,1)), axis=1)
                    test_losses = np.concatenate((test_losses,te_loss.reshape(-1,1)), axis=1)
        if (i_img_nr + 1)%self.batch_size != 0:
            if self.version!='old':
                self.update()
            if debug:
                tr_loss, te_loss = self.loss_per_level_all(instances, to)
                train_losses = np.concatenate((train_losses,tr_loss.reshape(-1,1)), axis=1)
                test_losses = np.concatenate((test_losses,te_loss.reshape(-1,1)), axis=1)
        if debug:
    	   return train_losses, test_losses
        
    def update(self):
        if self.version == 'multi':
            self.w_multi -= (self.eta * self.w_update)
            self.w_update = np.zeros((self.prune_tree_levels,self.n_features))
        else:
            for upd_lvl in self.w_update:
                self.w -= (self.eta * upd_lvl)
            self.w_update = np.zeros((self.prune_tree_levels,self.n_features))
        self.eta = self.eta * (1+self.eta0*self.gamma*self.samples_seen)**-1
        self.predictor = IEP.IEP(self.w, 'prediction')
        
    def learn_max(self, img_data, functions):
        level_preds, functions = self.predictor.get_iep_levels(img_data, functions)
        #print 'preds: ',img_data.img_nr, level_preds, ' y: ', img_data.y
        
        #print level_preds
        ind_max = level_preds.index(max(level_preds))
        upd, _ = self.learner.iep(img_data, [], ind_max)#functions[ind_max]
        return 2 * (self.predict(img_data) - img_data.y) * upd + 2 * self.alpha * self.w, functions
        
    def learn_mean(self, img_data, functions):
        level_preds = self.predict_ind(img_data)
        iep_levels, _ = self.learner.get_iep_levels(img_data, functions)
        return 2 * (np.array(level_preds) - img_data.y).reshape(-1,1) * iep_levels + 2 * self.alpha * self.w, functions

    #tested
    def learn_multi(self, img_data, functions):
        ret = np.zeros((self.prune_tree_levels,self.n_features))
        if len(img_data.levels) >= 10:
            print img_data.img_nr, img_data.y, len(img_data.levels)
        for level in range(self.prune_tree_levels):
            if level >= len(img_data.levels):
                continue
            predictor = IEP.IEP(self.w_multi[level], 'prediction')
            level_pred, _ = predictor.iep(img_data, [], level)
            iep_level, _ = self.learner.iep(img_data, functions, level)
            #print level, iep_level, img_data.y
            if self.n_features == 1:
                assert abs(iep_level-img_data.y) < 0.0001
            if len(img_data.levels) >= 10:
                print level, np.min(iep_level), np.max(iep_level)
            #a = (2 * (level_pred - img_data.y) * iep_level + 2 * self.alpha * self.w_multi[level])
            ret[level,:] = (2 * (level_pred - img_data.y) * iep_level + 2 * self.alpha * self.w_multi[level])
        return ret, functions


    def learn_old(self, img_data, functions):
        for level in img_data.levels:
            preds_level = self.predict_old(img_data, level)
            iep_level, _ = self.learner.iep(img_data, functions, level)
            self.w += (self.eta * ((preds_level - img_data.y) * -iep_level))#self.w -= (self.eta * (2*(preds_level - img_data.y)*iep_level))
            self.predictor = IEP.IEP(self.w, 'prediction')

    def learn_ind(self, img_data, functions):
        level_preds = self.predict_ind(img_data)
        iep_levels, _ = self.learner.get_iep_levels(img_data, [])
        return np.sum(2 * (np.array(level_preds) - img_data.y).reshape(-1,1) * iep_levels + 2 * self.alpha * self.w, axis=0), functions
