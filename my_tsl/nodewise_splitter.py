import numpy as np

def nodewise_index(dataset):
    """since the split is node-wise, return all indices
    along the temporal axit, up to the window
    eparation between train/val/test will be done by the masks"""
    idx = np.arange(len(dataset))
    return idx, idx