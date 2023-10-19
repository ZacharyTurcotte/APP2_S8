import numpy as np

matrix = np.array([[2,1,0],[1,2,0],[0,0,7]])

def get_eigvector(matrix):
    eigvector = np.linalg.eig(matrix)
    return [eigvector[0],eigvector[1]]

print("Done")