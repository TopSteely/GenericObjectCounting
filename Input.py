import os
import numpy as np

class Input:
    def __init__(self, mode, category):
        self.category = category
        if mode == 'grid':
            self.coord_path = 'bla'
            self.label_path = 'bla'
            self.feature_path = 'bla'
        elif mode == 'mscoco':
            self.coord_path = 'bla'
            self.label_path = 'bla'
            self.feature_path = 'bla'
        elif mode == 'trancos':
            self.coord_path = 'bla'
            self.label_path = 'bla'
            self.feature_path = 'bla'
        else:
            if mode == 'pascal':
                self.coord_path =  '/var/node436/local/tstahl/new_Resnet_features/2nd/coords/1-%s.csv'
                self.label_path =  '/var/node436/local/tstahl/Coords_prop_windows/Labels/Labels/%s_%s_partial.txt'
                self.feature_path = '/var/node436/local/tstahl/new_Resnet_features/2nd/1-%s'
            elif mode == 'dennis':
                self.feature_path = 'bla'
        
    
    
    def get_coords(self, img_nr):
        if os.path.isfile(self.coord_path%(format(img_nr, "06d"))):
                f = open(self.coord_path%(format(img_nr, "06d")), 'r')
        else:
            print 'warning, no ' + self.coord_path%(format(img_nr, "06d"))
            exit()
        boxes = []
        for line in f:
            tmp = line.split(',')
            coord = []
            for s in tmp:
                coord.append(float(s))
            boxes.append(coord)
        self.coords = boxes
        
    def get_features(self, img_nr):
        if os.path.isfile(self.feature_path%(format(img_nr, "06d"))):
            self.features = np.loadtxt(self.feature_path%(format(img_nr, "06d")), delimiter=',')
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
        