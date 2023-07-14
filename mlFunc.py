import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy
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
   # C = computeCovDiag(D,mu)
    return C
def znorm(DTR, DTE):
    mu_DTR = vcol(DTR.mean(1))
    std_DTR = vcol(DTR.std(1))

    DTR_z = (DTR - mu_DTR) / std_DTR
    DTE_z = (DTE - mu_DTR) / std_DTR
    return DTR_z, DTE_z

def gaussianize_features(DTR, TO_GAUSS):
    P = []
    for dIdx in range(DTR.shape[0]):
        DT = vcol(TO_GAUSS[dIdx, :])
        X = DTR[dIdx, :] < DT
        R = (X.sum(1) + 1) / (DTR.shape[1] + 2)
        P.append(scipy.stats.norm.ppf(R))
    return numpy.vstack(P)

def shuffle_dataset(D, L):
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    return D[:, idx], L[idx]
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
    #DTR = numpy.hstack(numpy.array(dList, dtype=numpy.float32))
    #DTR = numpy.hstack(numpy.array(dList, dtype=numpy.float32))
    #numpyArr = numpyArr.reshape((len(Dlist),10))
    #finalArray = numpyArr.transpose()
    #print(numpyArr,"\n\n########\n\n")
    labelpy = numpy.array(listLabel, dtype=int)
    D, L = shuffle_dataset(numpyArr.T, labelpy)
    return D.T, L

def bayes_error_plot(pArray, scores, labels, minCost=False, th=None):
    y = []
    for p in pArray:
        pi = 1.0 / (1.0 + numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1, th))
    return numpy.array(y)




def bayes_error_min_act_plot(D, LTE,  ylim):
    p = numpy.linspace(-3, 3, 21)
    plt.plot(p, bayes_error_plot(p, D, LTE, minCost=False), color='r', label='actDCF')
    plt.plot(p, bayes_error_plot(p, D, LTE, minCost=True), color='b', label='minDCF')
    plt.ylim(0, ylim)
    plt.ylabel('DCF')
    plt.legend(loc = 'upper left')
    plt.show()

def bayes_error_min_act_plot_compare(mvg, qlog,svm,gmm, mvg_l, qlog_l,svm_l, gmm_l, ylim):
    p = numpy.linspace(-3, 3, 21)
    plt.plot(p, bayes_error_plot(p, mvg, mvg_l, minCost=False), color='r', label='MVG Full - actDCF' )
    plt.plot(p, bayes_error_plot(p, mvg, mvg_l, minCost=True), color='r', label='MVG Full - minDCF', linestyle='dashed')
    plt.plot(p, bayes_error_plot(p, qlog, qlog_l, minCost=False), color='b', label='Q-Log-Reg - actDCF')
    plt.plot(p, bayes_error_plot(p, qlog, qlog_l, minCost=True), color='b', label='Q-Log-Reg - minDCF', linestyle='dashed')
    plt.plot(p, bayes_error_plot(p, svm, svm_l, minCost=False), color='y', label='SVM RBF - actDCF')
    plt.plot(p, bayes_error_plot(p, svm, svm_l, minCost=True), color='y', label='SVM RBF - minDCF', linestyle='dashed')
    plt.plot(p, bayes_error_plot(p, gmm, gmm_l, minCost=False), color='c', label='GMM - actDCF')
    plt.plot(p, bayes_error_plot(p, gmm, gmm_l, minCost=True), color='c', label='GMM - minDCF', linestyle='dashed')

    plt.ylim(0, ylim)
    plt.ylabel('DCF')
    plt.legend(loc = 'upper left')
    plt.show()



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

def computeCovDiag(M, muc):

    cov = numpy.dot((M-muc),(M-muc).T)/M.shape[1]
    diagCov = cov*numpy.identity(cov.shape[0])
    return (diagCov)



















def confusion_matrix(Lpred, LTE, k=2):
    # k = number of classes
    conf = numpy.zeros((k, k))
    for i in range(k):
        for j in range(k):
            conf[i][j] = ((Lpred == i) * (LTE == j)).sum()
    return conf


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -numpy.log(pi * Cfn) + numpy.log((1 - pi) * Cfp)
    P = scores > th
    return numpy.int32(P)


def confusion_matrix_binary(Lpred, LTE):
    C = numpy.zeros((2, 2))
    C[0, 0] = ((Lpred == 0) * (LTE == 0)).sum()
    C[0, 1] = ((Lpred == 0) * (LTE == 1)).sum()
    C[1, 0] = ((Lpred == 1) * (LTE == 0)).sum()
    C[1, 1] = ((Lpred == 1) * (LTE == 1)).sum()
    return C


def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    return pi * Cfn * fnr + (1 - pi) * Cfp * fpr


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi * Cfn, (1 - pi) * Cfp)


def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = confusion_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return min(dcfList)
