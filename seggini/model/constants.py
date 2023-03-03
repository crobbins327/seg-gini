from pathlib import Path
import os
import pandas as pd
import numpy as np

LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"
GNN_NODE_FEAT_IN = "gnn_node_feat_in"
GNN_NODE_FEAT_OUT = "gnn_node_feat_out"

# Total node classes including background/tissue
NODE_CLASSES = 5
# Total slide/graph classes
SLIDE_CLASSES = 4
# INCLUDE_CLASSES = [1, 2, 3, 4, 5]
# Include these classes in the weighting
INCLUDE_NODE_CLASSES = [0, 1, 2, 3, 4]
# Used to explain node classes relative to graph classification
EXPLAIN_NODE_CLASSES = [1, 2, 3, 4]
EXPLAIN_GRAPH_CLASSES = [0, 1, 2, 3]
BACKGROUND_CLASS = 0
VARIABLE_SIZE = True
WSI_FIX = True
THRESHOLD = 0.003
DISCARD_THRESHOLD = 5000
VALID_FOLDS = [0, 1, 2, 3]
MIN_COMBINED_COMP_NODES = 50

# MASK_VALUE_TO_TEXT = {
#     0: "unlabelled",
#     1: "tissue",
#     2: "neg",
#     3: "low",
#     4: "mod",
#     5: "hi"
# }
MASK_VALUE_TO_TEXT = {
    0: "unlabeled",
    # 1: "tissue",
    1: "neg",
    2: "low",
    3: "mod",
    4: "hi",
}
# MASK_VALUE_TO_COLOR = {0: "white", 1: "brown", 2: "blue", 3: "green", 4: "yellow", 5: "red"}
MASK_VALUE_TO_COLOR = {0: "white", 1: "blue", 2: "green", 3: "yellow", 4: "red"}


class Constants:
    def __init__(self, base_path: Path, mode: str, fold: int, partial: int):
        self.BASE_PATH = base_path
        self.MODE = mode
        self.FOLD = fold
        self.PARTIAL = partial

        assert fold in VALID_FOLDS, f"Fold must be in {VALID_FOLDS} but is {self.FOLD}"
        self.set_constants()

    def set_constants(self):
        self.PREPROCESS_PATH = os.path.join(self.BASE_PATH, 'preprocess')
        self.IMAGES_DF = os.path.join(self.BASE_PATH, 'pickles', 'images.pickle')
        self.ANNOTATIONS_DF = os.path.join(self.BASE_PATH, 'pickles', Path('annotation_masks_' + str(self.PARTIAL) + '.pickle'))
        self.LABELS_DF = os.path.join(self.BASE_PATH, 'pickles', 'image_level_annotations.pickle')

        # self.ID_PATHS = []
        self.ID_NAMES = np.array([])
        self.SPLIT_DIR = r"C:\Users\snibb\Projects\HS-HER2_MIL\preprocessing\splits\HS-HER2_pro_100"
        if self.MODE == 'train':
            self.ID_NAMES = pd.read_csv(os.path.join(self.SPLIT_DIR, "splits_{}.csv".format(self.FOLD)))["train"].values
            # self.ID_PATHS.append(os.path.join('partition' , 'Train' , f"Val{self.FOLD}" , "Train.csv"))
        elif self.MODE == 'val':
            self.ID_NAMES = pd.read_csv(os.path.join(self.SPLIT_DIR, "splits_{}.csv".format(self.FOLD)))["val"].values
            # self.ID_PATHS.append(os.path.join('partition' , 'Train' , f"Val{self.FOLD}" , "Val.csv"))
        elif self.MODE == 'test':
            self.ID_NAMES = pd.read_csv(os.path.join(self.SPLIT_DIR, "splits_{}.csv".format(self.FOLD)))["test"].values
            # self.ID_PATHS.append(os.path.join('partition' , 'Test' , "Test.csv"))
