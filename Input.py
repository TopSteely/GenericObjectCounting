import os
import numpy as np

class Input:
    def __init__(self, mode, category):
        self.mode = mode
        self.category = category
        if self.mode == 'grid':
            self.coord_path = 'bla'
            self.label_path = 'bla'
            self.feature_path = 'bla'
        elif self.mode == 'mscoco':
            self.coord_path = 'bla'
            self.label_path = 'bla'
            self.feature_path = 'bla'
        elif self.mode == 'trancos':
            self.coord_path = 'bla'
            self.label_path = 'bla'
            self.feature_path = 'bla'
        else:
            if self.mode == 'pascal':
                self.coord_path =  '/var/node436/local/tstahl/new_Resnet_features/2nd/coords/1-%s.csv'
                self.label_path =  '/var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/%s_%s_partial.txt'
                self.feature_path = '/var/node436/local/tstahl/new_Resnet_features/2nd/1-%s.csv'
                self.coord_tree_path = '/var/node436/local/tstahl/Coords_prop_windows/%s.txt'
            elif self.mode == 'dennis':
                self.feature_path = 'bla'
        training_numbers_tmp, self.test_numbers = self.get_training_numbers()
        self.training_numbers, self.val_numbers = self.get_val_numbers(training_numbers_tmp)
        self.category_train, self.category_val = self.get_category_imgs()
        
    def get_category_imgs(self):
        tr_images = []
        te_images = []
        for img in self.training_numbers:
            y = self.get_label(img)
            if y > 0:
                tr_images.append(img)
        for img in self.val_numbers:
            y = self.get_label(img)
            if y > 0:
                te_images.append(img)
        return tr_images, te_images
                
    def get_training_numbers(self):     
        if self.mode == 'pascal':
            file = open('/var/scratch/tstahl/IO/test.txt')
            test_imgs = []
            train_imgs = []
            for line in file:
                test_imgs.append(int(line))
            for i in range(1,9963):
                if i not in test_imgs:
                    train_imgs.append(i)
            return test_imgs, train_imgs
        
    def get_val_numbers(self, train_imgs):
        if self.mode == 'pascal':
            file = open('/var/scratch/tstahl/IO/val.txt', 'r')
            eval_images = []
            for line in file:
                im_nr = int(line)
                eval_images.append(im_nr)
            return [x for x in train_imgs if x not in eval_images], eval_images
    
    def get_coords(self, img_nr):
        if os.path.isfile(self.coord_path%(format(img_nr, "06d"))):
            ret = np.loadtxt(self.coord_path%(format(img_nr, "06d")), delimiter=',')
            print len(ret), type(ret[0])
            if isinstance(ret[0], np.float64):
                return [ret]
            else:
                return ret
            
            
    def get_coords_tree(self, img_nr):
        if os.path.isfile(self.coord_tree_path%(format(img_nr, "06d"))):
            f = open(self.coord_tree_path%(format(img_nr, "06d")), 'r')
        else:
            print 'warning, no ' + self.coord_tree_path%(format(img_nr, "06d"))
            exit()
        boxes = []
        for line in f:
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append(coord)
        return boxes
        
    def get_features(self, img_nr):
        if os.path.isfile(self.feature_path%(format(img_nr, "06d"))):
            return np.loadtxt(self.feature_path%(format(img_nr, "06d")), delimiter=',')
        else:
             print 'warning ' + self.feature_path%(format(img_nr, "06d")) + 'does not exist'
             exit()
        

    def get_label(self, img_nr):
        if os.path.isfile(self.label_path%(format(img_nr, "06d"),self.category)):
            file = open(self.label_path%(format(img_nr, "06d"),self.category), 'r')
        else:
            print 'warning ' + (self.label_path%(format(img_nr, "06d"),self.category)) + ' does not exist '
            exit()
        line = file.readline()
        tmp = line.split()[0]
        label = float(tmp)
        return label