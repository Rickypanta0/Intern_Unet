# src/data_loader.py
"""
Moduli per il caricamento e preprocessing dei dati.
"""
import os
from typing import List, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt


folds = [os.path.join('data', 'raw', 'Fold 2', 'masks', 'binary_masks.npy'),
         os.path.join('data', 'raw', 'Fold 3', 'masks', 'masks.npy'),
        os.path.join('data', 'raw', 'Fold 1', 'masks', 'masks.npy')] 

masks = np.load(folds[0], mmap_mode='r')
#for f in masks:
#    plt.imshow(f)
#    plt.show()



