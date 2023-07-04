import numpy
from scipy.optimize import fmin_l_bfgs_b
from mlFunc import *
from itertools import repeat

def bayes_error_plot_compare(pi, scores, labels):
    y = []
#    pi = 1.0 / (1.0 + numpy.exp(-pi)) #todo
    y.append(compute_min_DCF(scores, labels, pi, 1, 1))
    return numpy.array(y)

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
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return numpy.array(dcfList).min()

def plot_DCF(x, y, xlabel, title, base=10):
    plt.figure()
    plt.plot(x, y[0], label= 'min DCF prior=0.5', color='b')
    plt.plot(x, y[1], label= 'min DCF prior=0.9', color='g')
    plt.plot(x, y[2], label= 'min DCF prior=0.1', color='r')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend([ "min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    #plt.savefig('./images/DCF_' + title+ '.svg')
    plt.show()
    return

def calculate_lbgf(H, DTR, C):
    def JDual(alpha):
        Ha = numpy.dot(H, vcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, _x, _y = fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=1.0,
        maxiter=10000,
        maxfun=100000,
    )

    return alphaStar, JDual, LDual

class SupportVectorMachine:
    def __init__(self):
        self.w = []
        self.pl = []
        self.dl = []
        self.dg = []

    def train_SVM_linear(self, DTR, LTR, C, K):
        DTREXT = numpy.vstack([DTR, K * numpy.ones((1, DTR.shape[1]))])
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1

        H = numpy.dot(DTREXT.T, DTREXT)
        H = vcol(Z) * vrow(Z) * H

        def JPrimal(w):
            S = numpy.dot(vrow(w), DTREXT)
            loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z * S).sum()
            return 0.5 * numpy.linalg.norm(w) ** 2 + C * loss

        alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)
        wStar = numpy.dot(DTREXT, vcol(alphaStar) * vcol(Z))
        return wStar, JPrimal(wStar)
    
    def train_SVM_polynomial(self, DTR, LTR, C, K=1, constant=0, degree=2):
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1

        H = (numpy.dot(DTR.T, DTR) + constant) ** degree + K ** 2
        # Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
        # H = numpy.exp(-Dist)
        H = vcol(Z) * vrow(Z) * H

        alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

        return alphaStar, JDual(alphaStar)[0]
    
    def train_SVM_RBF(self, DTR, LTR, C, K=1, gamma=1.):
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1

        # kernel function
        kernel = numpy.zeros((DTR.shape[1], DTR.shape[1]))
        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                kernel[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
        H = vcol(Z) * vrow(Z) * kernel

        alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

        return alphaStar, JDual(alphaStar)[0]

    def predict_SVM_Linear(self, D, L, C, K, Dte):
        scoresLin_append = []
        wStar, primal = self.train_SVM_linear(D, L, C, K)
        DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])

        scores = numpy.dot(wStar.T, DTEEXT).ravel()
        scoresLin_append.append(scores)
        return scoresLin_append
    
    def predict_SVM_Pol(self, D, L, C, K, Dte, costant, degree):
        scoresPol_append = []
        #costant = 0
        #degree = 2
        aStar, primal = self.train_SVM_polynomial(D, L, C, K, costant, degree)
        Z = numpy.zeros(L.shape)
        Z[L == 1] = 1
        Z[L == 0] = -1
        kernel = (numpy.dot(D.T, Dte) + costant) ** degree + K * K
        scores = numpy.sum(numpy.dot(aStar * vrow(Z), kernel), axis=0)
        scoresPol_append.append(scores)
        return scoresPol_append
    
    def predict_SVM_RBF(self, D, L, C, K, Dte, gamma):
        scoresRBF_append = []
        Z = L * 2 - 1
        #gamma=0.001

        aStar, loss = self.train_SVM_RBF(D, L, C, K, gamma)
        kern = numpy.zeros((D.shape[1], Dte.shape[1]))
        for i in range(D.shape[1]):
            for j in range(Dte.shape[1]):
                kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
        scores = numpy.sum(numpy.dot(aStar * vrow(Z), kern), axis=0)
        scoresRBF_append.append(scores)
        return scoresRBF_append

        