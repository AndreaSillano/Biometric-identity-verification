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


def computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(pi1, Cfn, Cfp, llrs, labels, t):
    # Now, if the llr is > than the threshold => predicted class is 1
    # If the llr is <= than the threshold => predicted class is 0
    predictedLabels = (llrs > t).astype(int)
    # Compute the confusion matrix
    m = confusionMatrixV2THRESHOLD(predictedLabels, labels, 2, pi1, Cfn, Cfp)
    return m

def confusionMatrixV2THRESHOLD(pl, LEV, K, pi1, Cfn, Cfp):
    # Initialize matrix of K x K zeros
    matrix = numpy.zeros((K, K))
    # Here we're not talking about costs yet! We're only computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(LEV.size):
        # Update "counter" in proper position
        matrix[pl[i], LEV[i]] += 1
    return matrix

def evaluationBinaryTaskV2THRESHOLD(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    # Compute empirical Bayes risk, that is the cost that we pay due to our
    # decisions c* for the test data.
    DCFu = pi1*Cfn*FNR+(1-pi1)*Cfp*FPR
    return (DCFu, FNR, FPR)

def normalizedEvaluationBinaryTaskV2THRESHOLD(pi1, Cfn, Cfp, DCFu):
    # Define vector with dummy costs
    dummyCosts = numpy.array([pi1*Cfn, (1-pi1)*Cfp])
    # Compute risk for an optimal dummy system
    index = numpy.argmin(dummyCosts)
    # Compute normalized DCF
    DCFn = DCFu/dummyCosts[index]
    return DCFn


def computeOptimalBayesDecisionBinaryTask(pi1, Cfn, Cfp, llrs, labels):
    # Compute the threshold
    t = -numpy.log((pi1*Cfn)/((1-pi1)*Cfp))
    # Now, if the llr is > than the threshold => predicted class is 1
    # If the llr is <= than the threshold => predicted class is 0
    predictedLabels = (llrs > t).astype(int)
    # Compute the confusion matrix
    m = confusionMatrix(predictedLabels, labels, 2, pi1, Cfn, Cfp)
    return m

def confusionMatrix(pl, LEV, K, pi1, Cfn, Cfp):
    # Initialize matrix of K x K zeros
    matrix = numpy.zeros((K, K))
    # Here we're not talking about costs yet! We're only computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(LEV.size):
        # Update "counter" in proper position
        matrix[pl[i], LEV[i]] += 1
    print("Confusion matrix with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f:" %
          (pi1, Cfn, Cfp))
    print(matrix)
    return matrix

def evaluationBinaryTask(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    # Compute empirical Bayes risk, that is the cost that we pay due to our
    # decisions c* for the test data.
    DCFu = pi1*Cfn*FNR+(1-pi1)*Cfp*FPR
    print("Unnormalized detection cost function with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: %.3f" %
          (pi1, Cfn, Cfp, DCFu))
    return DCFu

def normalizedEvaluationBinaryTask(pi1, Cfn, Cfp, DCFu):
    # Define vector with dummy costs
    dummyCosts = numpy.array([pi1*Cfn, (1-pi1)*Cfp])
    # Compute risk for an optimal dummy system
    index = numpy.argmin(dummyCosts)
    # Compute normalized DCF
    DCFn = DCFu/dummyCosts[index]
    print("Normalized detection cost function with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: %.3f" %
          (pi1, Cfn, Cfp, DCFn))
    return

def evaluation(scores, LTE, pi,C_fn, C_fp):

    DFCList =[]


    m1 = computeOptimalBayesDecisionBinaryTask(pi, C_fn, C_fp, scores, LTE)
    DCFu1 = evaluationBinaryTask(pi, C_fn, C_fp, m1)
    normalizedEvaluationBinaryTask(pi, C_fn, C_fp, DCFu1)

    #for t in th:
    #    m1 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
    #        pi, C_fn, C_fp, scores, LTE, t)
    #    (DCFu1, _, _) = evaluationBinaryTaskV2THRESHOLD(pi, C_fn, C_fp, m1)
    #    DFCList.append(normalizedEvaluationBinaryTaskV2THRESHOLD(pi, C_fn, C_fp, DCFu1))

    return DCFu1

def compute_min_DCF1(scores, pi,C_fn, C_fp, LTR):
    th = numpy.sort(scores)
    DFCList =[]
    for t in th:
        m1 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            pi, C_fn, C_fp, scores, LTR, t)
        (DCFu1, _, _) = evaluationBinaryTaskV2THRESHOLD(pi, C_fn, C_fp, m1)
        DFCList.append(normalizedEvaluationBinaryTaskV2THRESHOLD(pi, C_fn, C_fp, DCFu1))

    index = np.argmin(DFCList)
    print("Min DCF with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: " %
          (0.5, 1, 1))
    print(DCFarr1[index])





















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
    return numpy.array(dcfList).min()
