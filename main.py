

import matplotlib.pyplot as plt
import numpy
from plotter import Plotter
from dimensionality_reduction import DimensionalityReduction
from validator import Validation
from evaluator import Evaluator
from mlFunc import *

def validation(DTR,LTR, VA, dimRed):
    print("##################################")
    print("VALIDATION")
    print("##################################")
    print("***MVG VALIDATION***")
    VA.MVG_validation(DTR,LTR, 0.5, 1,10)
    for i in range (7,10):
         print("PCA con", i)
         DPA = dimRed.PCA(DTR, i)
         VA.MVG_validation(DPA.T, LTR, 0.5, 1, 10)
    print("***LR VALIDATION***")
    VA.LR_validation(DTR, LTR, 0.5, 1, 10, plot=True)
    for i in range (7,10):
         print("PCA con", i)
         DPA = dimRed.PCA(DTR, i)
         VA.LR_validation(DPA.T,LTR, 0.1,1,10, False)

    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    for k in K_arr:
        for c in C_arr:
            print("SVM, K: ",k," C: ", c)
            VA.SVM_validation(DTR, LTR, 0.9, 1, 10, k, c)
    VA.SVM_validation(DTR, LTR, 0.5, 1, 10, 10, 10, False)

    VA.GMM_validation(DTR,LTR, 0.5,1,10, 1,4, 0.1, 0.01, True)
    for i in range (7,10):
         print("PCA con", i)
         DPA = dimRed.PCA(DTR, i)
         VA.GMM_validation(DPA.T,LTR, 0.5,1,10, 1,4, 0.1, 0.01)

    VA.plot_ROC(DTR, LTR, 0.5)
    print("Comparing DCF")
    VA.plot_minDCF_cal_score(DTR, LTR, 0.5)
    return

def evaluation(DTE, LTE, DTR, LTR, EV,  dimRed):
    print("##################################")
    print("EVALUATION")
    print("##################################")
    print("***MVG EVALUATION***")
    EV.MVG_evaluation(DTE, LTE, DTR, LTR, 0.5, 1, 10)
    for i in range(7, 10):
       print("PCA con", i)
       DPA = dimRed.PCA(DTR, i)
       DPE = dimRed.PCA_DTE(DTR, i,DTE)
       EV.MVG_evaluation(DPE.T, LTE, DPA.T, LTR, 0.5, 1, 10)

    print("***LR EVALUATION***")

    EV.LR_evaluation(DTE, LTE, DTR, LTR, 0.1, 1, 10, False)
    for i in range(7, 10):
        print("PCA con", i)
        DPA = dimRed.PCA(DTR, i)
        DPE = dimRed.PCA_DTE(DTR, i,DTE)
        EV.LR_evaluation(DPE.T, LTE, DPA.T, LTR, 0.5, 1, 10)
    print("***SVM EVALUATION***")
    EV.SVM_evaluation(DTE.T, LTE, DTR.T, LTR, 0.5, 1, 10)
    for i in range(7, 10):
       print("PCA con", i)
       DPA = dimRed.PCA(DTR, i)
       DPE = dimRed.PCA_DTE(DTR, i,DTE)
       EV.SVM_evaluation(DPE, LTE, DPA, LTR, 0.9, 1, 10)

    print("***GMM EVALUATION***")
    EV.GMM_evaluation(DTE, LTE, DTR, LTR, 0.9, 1, 10, 1, 16, 0.1, 0.01)
    for i in range(7, 10):
        print("PCA con", i)
        DPA = dimRed.PCA(DTR, i)
        DPE = dimRed.PCA_DTE(DTR, i,DTE)
        EV.GMM_evaluation(DPE.T, LTE, DPA.T, LTR, 0.9, 1, 10, 1, 16, 0.1, 0.01)

    EV.plot_ROC(DTR, LTR, DTE,LTE,0.5)
    print("Comparing DCF")
    EV.plot_minDCF_cal_score(DTR, LTR, DTE,LTE,0.5)

if __name__ == "__main__":

    DTR,LTR = load("Train.txt")

    DTE, LTE = load("Test.txt")


    print("TRAINING AUTHENTIC: ", DTR.T[:,LTR==1].shape[1])
    print("TRAINING SPOFFED", DTR.T[:, LTR==0].shape[1])

    print("TEST AUTHENTIC: ", DTE.T[:, LTE == 1].shape[1])
    print("TEST SPOFFED", DTE.T[:, LTE == 0].shape[1])

    plt = Plotter()
    dimRed = DimensionalityReduction()
    VA = Validation()
    EV = Evaluator()

    plt.plot_correlations(DTR.T, "heatmap")
    plt.plot_correlations(DTR.T[:, LTR == 0], "heatmap_spoofed_", cmap="Reds")
    plt.plot_correlations(DTR.T[:, LTR == 1], "heatmap_authentic_", cmap="Blues")

    validation(DTR, LTR, VA, dimRed)
    evaluation(DTE, LTE, DTR, LTR, EV, dimRed)





