#fichier avec des fonction qui peuvent être utile

import cv2
import numpy as np


def distance_n_dim(data,dim):
    res = 0
    for i in range(dim):
        res = data[i]**2 + res

    return np.sqrt(res)