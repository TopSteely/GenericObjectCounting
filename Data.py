from utils import create_tree_as_extracted, surface_area_old, sort_boxes
import Input

class Data:
    def __init__(self, load, img_nr, prune_tree_levels):
        print img_nr
        self.img_nr = img_nr
        self.boxes = load.get_coords(img_nr)
        self.X = load.get_features(img_nr)
        self.y = load.get_label(img_nr)
        self.tree_boxes = load.get_coords_tree(img_nr)
        self.tree_boxes = sort_boxes(self.tree_boxes)
        self.G, levels = create_tree_as_extracted(self.tree_boxes)
        print len(self.G.nodes())
        print self.G.edges()
        #prune tree to only have levels which fully cover the image, tested
        total_size = surface_area_old(self.tree_boxes, levels[0])
        for level in levels:
            sa = surface_area_old(self.tree_boxes, levels[level])
            sa_co = sa/total_size
            print level, sa_co
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
        print self.G.edges()
            
    def scale(self, scaler):
        return scaler.transform(self.X)