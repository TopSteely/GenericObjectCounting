from utils import create_tree, surface_area
from Input import get_coords, get_features, get_label

class Data:
    def __init__(self, Input_, img_nr, prune_tree_levels):
        self.img_nr = img_nr
        self.boxes = Input_.get_coords(img_nr)
        self.X = Input_.get_features(img_nr)
        self.y = Input_.get_label(img_nr)
        
        self.G, levels = create_tree(self.boxes)
        #prune tree to only have levels which fully cover the image, tested
        total_size = surface_area(self.boxes, levels[0])
        for level in levels:
            sa = surface_area(self.boxes, levels[level])
            sa_co = sa/total_size
            if sa_co != 1.0:
                self.G.remove_nodes_from(levels[level])
            else:
                nr_levels_covered = level
        levels = {k: levels[k] for k in range(0,nr_levels_covered + 1)}
        # prune levels, speedup + performance 
        levels_tmp = {k:v for k,v in levels.iteritems() if k<prune_tree_levels}
        levels_gone = {k:v for k,v in levels.iteritems() if k>=prune_tree_levels}
        self.levels = levels_tmp
        #prune tree as well, for patches training
        for trash_level in levels_gone.values():
            self.G.remove_nodes_from(trash_level)