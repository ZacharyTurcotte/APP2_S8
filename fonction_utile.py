#fichier avec des fonction qui peuvent être utile

import numpy as np
from skimage import io as skiio
import os


def distance_n_dim(data,dim):
    res = 0
    for i in range(dim):
        res = data[i]**2 + res

    return np.sqrt(res)