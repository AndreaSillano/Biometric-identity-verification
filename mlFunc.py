import numpy as numpy
import matplotlib.pyplot as plt
import math

def vcol(vett):
    return vett.reshape(vett.size, 1)

def vrow(vett):
    return vett.reshape(1, vett.size)

def empirical_mean(D):
    return vcol(D.mean(1))


def empirical_covariance(D, mu):
    n = numpy.shape(D)[1]
    DC = D - vcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    return C

def load(name):
    try:
        file = open(name, "r")
    except FileNotFoundError:
        exit(-1)

    Dlist = []
    listLabel = []
    for row in file:
        line = row.rstrip().split(",")
        singleLine = line[0:10]
        label = line[-1]
        
        Dlist.append(singleLine)
        listLabel.append(label)
    
    numpyArr = numpy.array([[Dlist]], dtype=float)
    #numpyFlowers = numpyArr.reshape((150,4))
    #finalFlowers = numpyFlowers.transpose()
    print(numpyArr,"\n\n########\n\n")
    labelpy = numpy.array(listLabel)
    print(labelpy)

    return (numpyArr, labelpy)

def randomize(D, L, seed=0):
    nTrain = int(D.shape[1])
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    
    DTR = D[:, idxTrain]
    LTR = L[idxTrain]
    
    return DTR, LTR