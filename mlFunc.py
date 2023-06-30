import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib
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
    
    numpyArr = numpy.array(Dlist, dtype=float)
    #numpyArr = numpyArr.reshape((len(Dlist),10))
    #finalArray = numpyArr.transpose()
    #print(numpyArr,"\n\n########\n\n")
    labelpy = numpy.array(listLabel, dtype=int)

    return (numpyArr, labelpy)

def randomize(D, L, seed=0):
    nTrain = int(D.shape[1])
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    
    DTR = D[:, idxTrain]
    LTR = L[idxTrain]
    
    return DTR, LTR

def compute_correlation(X, Y):
    x_sum = numpy.sum(X)
    y_sum = numpy.sum(Y)

    x2_sum = numpy.sum(X ** 2)
    y2_sum = numpy.sum(Y ** 2)

    sum_cross_prod = numpy.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = numpy.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr


def plot_correlations(DTR, title, cmap="Greys"):
    corr = numpy.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem

    plt.rcParams['axes.linewidth'] = 0.2

    # Creazione della heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(numpy.abs(corr), cmap=cmap, aspect='equal')

    # Personalizzazioni dell'asse x e y
    ax.set_xticks(numpy.arange(corr.shape[1]))
    ax.set_yticks(numpy.arange(corr.shape[0]))
    ax.set_xticklabels(numpy.arange(corr.shape[1]))
    ax.set_yticklabels(numpy.arange(corr.shape[0]))
    ax.tick_params(axis='both', which='both', length=0)

    # Aggiunta della barra dei colori
    cbar = plt.colorbar(heatmap)

    # Mostra il grafico
    plt.show()
    #fig = heatmap.get_figure()
    #fig.savefig("./images/" + title + ".svg")

def computeCovDiag(M, muc):

    cov = numpy.dot((M-muc),(M-muc).T)/M.shape[1]
    diagCov = numpy.diag(numpy.diag(cov))
    return (diagCov)

def evaluation(scores, labels, th, C_fn, C_fp):
    pred_labels = int(scores > th)

    confusion_matrix = numpy.zeros(2,2)
    for i in range(len(labels)):
        confusion_matrix[labels[i], pred_labels[i]] += 1

    FN = confusion_matrix[1, 0]
    TP = confusion_matrix[1, 1]
    FNR = FN / (FN + TP)

    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]
    FPR = FP / (FP + TN)
    DFC = C_fn * FNR + C_fp * FPR
    return DFC