import numpy as np
import copy
def flip_x(X, Y):
    return np.flip(copy.deepcopy(X), axis = 1), np.flip(copy.deepcopy(Y), axis = 1)

def flip_y(X, Y):
    return np.flip(copy.deepcopy(X), axis = 2), np.flip(copy.deepcopy(Y), axis = 2)

def identity(X, Y):
    return X, Y


def augmentate(X,Y):
    list_X, list_Y =[],[]
    for modif in [identity, flip_x, flip_y]:
        X_modif, Y_modif = modif(X,Y)
        list_X.append(X_modif)
        list_Y.append(Y_modif)
    return np.concatenate(list_X, axis = 0), np.concatenate(list_Y, axis = 0)

if __name__ == '__main__':
    X = np.zeros((10,96,96,1))
    Y = np.zeros((10,96,96,1))
    X, Y = augmentate(X,Y)