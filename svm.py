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


def train_SVM_linear(DTR, LTR, C, K=1):
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

class SupportVectorMachine:
    def __init__(self):
        self.w = []
        self.pl = []
        self.dl = []
        self.dg = []

    def validation_SVM(self, DTR, LTR, K_arr, C_arr, appendToTitle):
        for K in K_arr:
            for C in C_arr:
                self.kfold_SVM(DTR, LTR, K, C, appendToTitle)

        print("codio")
        x = numpy.logspace(-4, 1, 10)
        y = numpy.array([])
        y_05 = numpy.array([])
        y_09 = numpy.array([])
        y_01 = numpy.array([])
        print(x,"+",len(x))
        for xi in x:
            print(xi)
            scores, labels = self.kfold_SVM_calibration(DTR, LTR, 1.0, xi)
            y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
            y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
            y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))

        y = numpy.hstack((y, y_05))
        y = numpy.vstack((y, y_09))
        y = numpy.vstack((y, y_01))

        plot_DCF(x, y, 'C', appendToTitle + 'SVM_minDCF_comparison')
    
    def kfold_SVM(self, DTR, LTR, K, C, appendToTitle, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
        k = 5
        Dtr = numpy.split(DTR, k, axis=1)
        Ltr = numpy.split(LTR, k)

        scores_append = []
        PCA_SVM_scores_append = []
        PCA2_SVM_scores_append = []
        SVM_labels = []

        for i in range(k):
            D = []
            L = []
            if i == 0:
                D.append(numpy.hstack(Dtr[i + 1:]))
                L.append(numpy.hstack(Ltr[i + 1:]))
            elif i == k - 1:
                D.append(numpy.hstack(Dtr[:i]))
                L.append(numpy.hstack(Ltr[:i]))
            else:
                D.append(numpy.hstack(Dtr[:i]))
                D.append(numpy.hstack(Dtr[i + 1:]))
                L.append(numpy.hstack(Ltr[:i]))
                L.append(numpy.hstack(Ltr[i + 1:]))

            D = numpy.hstack(D)
            L = numpy.hstack(L)

            Dte = Dtr[i]
            Lte = Ltr[i]

            print(i)
            wStar, primal = train_SVM_linear(D, L, C=C, K=K)
            DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])

            scores = numpy.dot(wStar.T, DTEEXT).ravel()
            scores_append.append(scores)

            SVM_labels = numpy.append(SVM_labels, Lte, axis=0)
            SVM_labels = numpy.hstack(SVM_labels)


        scores_append = numpy.hstack(scores_append)
        scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.5, 1, 1)
        print(scores_tot)

        # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

        # Cfn and Ctp are set to 1
        #bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)

        ###############################

        # π = 0.1
        scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.1, 1, 1)
        print(scores_tot)
        ###############################

        # π = 0.9
        scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.9, 1, 1)
        print(scores_tot)

    def kfold_SVM_calibration(self, DTR, LTR, K, C, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
        k = 5
        Dtr = numpy.split(DTR, k, axis=1)
        Ltr = numpy.split(LTR, k)

        scores_append = []
        LR_labels = []

        for i in range(k):
            D = []
            L = []
            if i == 0:
                D.append(numpy.hstack(Dtr[i + 1:]))
                L.append(numpy.hstack(Ltr[i + 1:]))
            elif i == k - 1:
                D.append(numpy.hstack(Dtr[:i]))
                L.append(numpy.hstack(Ltr[:i]))
            else:
                D.append(numpy.hstack(Dtr[:i]))
                D.append(numpy.hstack(Dtr[i + 1:]))
                L.append(numpy.hstack(Ltr[:i]))
                L.append(numpy.hstack(Ltr[i + 1:]))

            D = numpy.hstack(D)
            L = numpy.hstack(L)

            Dte = Dtr[i]
            Lte = Ltr[i]

            print(i)
            wStar, primal = train_SVM_linear(D, L, C=C, K=K)
            DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])

            scores = numpy.dot(wStar.T, DTEEXT).ravel()
            scores_append.append(scores)

            LR_labels = numpy.append(LR_labels, Lte, axis=0)
            LR_labels = numpy.hstack(LR_labels)

        return numpy.hstack(scores_append), LR_labels
    
    def evaluate_SVM(self, DTR, LTR, DTE, LTE, K, C, appendToTitle, PCA_Flag=True):
        scores_append = []
        SVM_labels = []

        wStar, _ = train_SVM_linear(DTR, LTR, C=C, K=K)

        DTEEXT = numpy.vstack([DTE, K * numpy.ones((1, DTE.shape[1]))])

        scores = numpy.dot(wStar.T, DTEEXT).ravel()
        scores_append.append(scores)

        SVM_labels = numpy.append(SVM_labels, LTE, axis=0)
        SVM_labels = numpy.hstack(SVM_labels)

        scores_append = numpy.hstack(scores_append)
        scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.5, 1, 1)

        # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

        # Cfn and Ctp are set to 1
        # bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)


        ###############################

        # π = 0.1
        scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.1, 1, 1)

        ###############################

        # π = 0.9
        scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.9, 1, 1)


    def svm_tuning(self, DTR, LTR,DTE, LTE, K, C):
        scores_append = []
        labels = []

        wStar, _ = train_SVM_linear(DTR, LTR, C=C, K=K)
        DTEEXT = numpy.vstack([DTE, K * numpy.ones((1, DTE.shape[1]))])

        scores = numpy.dot(wStar.T, DTEEXT).ravel()
        scores_append.append(scores)

        labels = numpy.append(labels, LTE, axis=0)
        labels = numpy.hstack(labels)

        return numpy.hstack(scores_append), labels


    def evaluation_SVM(self, DTR, LTR, DTE, LTE, K_arr, C_arr, appendToTitle, PCA_Flag=True):
        for K in K_arr:
            for C in C_arr:
                self.evaluate_SVM(DTR, LTR, DTE, LTE, K, C, appendToTitle, PCA_Flag=False)
        x = numpy.logspace(-3, 2, 6)
        y = numpy.array([])
        y_05 = numpy.array([])
        y_09 = numpy.array([])
        y_01 = numpy.array([])
        for xi in x:
            scores, labels = self.svm_tuning(DTR, LTR, DTE, LTE, 1.0, xi)
            y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
            y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
            y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))

        y = numpy.hstack((y, y_05))
        y = numpy.vstack((y, y_09))
        y = numpy.vstack((y, y_01))

        plot_DCF(x, y, 'C', appendToTitle + 'Linear_SVM_minDCF_comparison')