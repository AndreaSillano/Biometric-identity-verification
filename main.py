

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
    #DPA = dimRed.PCA(DTR, 8)
    #VA.MVG_validation(DPA.T,LTR, 0.5, 1,10)
    # for i in range (7,10):
    #      print("PCA con", i)
    #      DPA = dimRed.PCA(DTR, i)
    #      VA.MVG_validation(DPA.T, LTR, 0.5, 1, 10)
    #DPA = dimRed.PCA(DTR, 7)
    #VA.LR_validation(DPA.T, LTR, 0.5, 1, 10, plot=True)
    # for i in range (7,10):
    #      print("PCA con", i)
    #      DPA = dimRed.PCA(DTR, i)
    #      VA.LR_validation(DPA.T,LTR, 0.1,1,10, False)

    # K_arr = [0.1, 1.0, 10.0]
    # C_arr = [0.01, 0.1, 1.0, 10.0]
    # for k in K_arr:
    #     for c in C_arr:
    #         print("SVM, K: ",k," C: ", c)
    #         VA.SVM_validation(DTR, LTR, 0.9, 1, 10, k, c)
    #VA.SVM_validation(DTR, LTR, 0.5, 1, 10, 10, 10, False)

    #VA.GMM_validation(DTR,LTR, 0.5,1,10, 1,8, 0.1, 0.01, True)
    # for i in range (7,10):
    #      print("PCA con", i)
    #      DPA = dimRed.PCA(DTR, i)
    #      VA.GMM_validation(DPA.T,LTR, 0.5,1,10, 1,8, 0.1, 0.01)
    # DPA_7 = dimRed.PCA(DTR, 7)
    # VA.GMM_validation(DPA_7.T,LTR, 0.5,1,10, 1,4, 0.1, 0.01, True)
    VA.plot_ROC(DTR, LTR, 0.5)
    #VA.GMM_validation(DPA_7.T,LTR, 0.5,1,10, 1,4, 0.1, 0.01, True)
    print("Comparing Scores")
    #VA.plot_minDCF_cal_score(DTR, LTR, 0.5)
    return

def evaluation(DTE, LTE, DTR, LTR, EV,  dimRed):
    print("##################################")
    print("EVALUATION")
    print("##################################")
    print("MVG EVALUATION")
    # #EV.MVG_evaluation(DTE, LTE, DTR, LTR, 0.5, 1, 10)
    # #for i in range(7, 10):
    # #    print("PCA con", i)
    # #    DPA = dimRed.PCA(DTR, i)
    # #    DPE = dimRed.PCA_DTE(DTR, i,DTE)
    # #    EV.MVG_evaluation(DPE.T, LTE, DPA.T, LTR, 0.5, 1, 10)

    print("LOGISTIC EVALUATION")

    #EV.LR_evaluation(DTE, LTE, DTR, LTR, 0.1, 1, 10, False)
    # for i in range(7, 10):
    #     print("PCA con", i)
    #     DPA = dimRed.PCA(DTR, i)
    #     DPE = dimRed.PCA_DTE(DTR, i,DTE)
    #     EV.LR_evaluation(DPE.T, LTE, DPA.T, LTR, 0.5, 1, 10)


    #EV.GMM_evaluation(DTE, LTE, DTR, LTR, 0.5, 1, 10, 1, 4, 0.1, 0.01)
    EV.SVM_evaluation(DTE.T, LTE, DTR.T, LTR, 0.5, 1, 10)
    #for i in range(7, 10):
     #   print("PCA con", i)
      #  DPA = dimRed.PCA(DTR, i)
       # DPE = dimRed.PCA_DTE(DTR, i,DTE)
        #EV.SVM_evaluation(DPE, LTE, DPA, LTR, 0.9, 1, 10)
    print("GMM")
    # EV.GMM_evaluation(DTE, LTE, DTR, LTR, 0.9, 1, 10, 1, 16, 0.1, 0.01)
    # for i in range(7, 10):
    #     print("PCA con", i)
    #     DPA = dimRed.PCA(DTR, i)
    #     DPE = dimRed.PCA_DTE(DTR, i,DTE)
    #     EV.GMM_evaluation(DPE.T, LTE, DPA.T, LTR, 0.9, 1, 10, 1, 16, 0.1, 0.01)
    EV.plot_minDCF_cal_score(DTR, LTR, DTE,LTE,0.5)

if __name__ == "__main__":

    DTR,LTR = load("Train.txt")

    DTE, LTE = load("Test.txt")


    #print(empirical_covariance(DTR.T[:,LTR==0], empirical_mean(DTR.T)))
    #print("TARGET\n")
    #print(empirical_covariance(DTR.T[:, LTR == 1], empirical_mean(DTR.T)))

    print("TRAINING AUTHENTIC: ", DTR.T[:,LTR==1].shape[1])
    print("TRAINING SPOFFED", DTR.T[:, LTR==0].shape[1])

    print("TEST AUTHENTIC: ", DTE.T[:, LTE == 1].shape[1])
    print("TEST SPOFFED", DTE.T[:, LTE == 0].shape[1])

    plt = Plotter()
    dimRed = DimensionalityReduction()
    VA = Validation()
    EV = Evaluator()

    #plt.plot_histogram(DTE,LTE)
    #plt.plot_histogram(DTR, LTR)
    #plt.plot_scatter(DTR, LTR)

    print("---------------PRINCIPAL COMPONENT ANALYSIS-------------")
    DPA = dimRed.PCA(DTR, 7)
    #DPEA = dimRed.PCA(DTR, 2)
    plt.plot_PCA_scatter(DPA,LTR)
    dimRed.evaluatePCA(DPA,LTR)
    print("---------------LINEAR DISCRIMINANT ANALYSIS-------------")
    DP = dimRed.LDA(DTR,LTR)
    DPE = dimRed.LDA(DTE,LTE)
    plt.plot_LDA_scatter(DP,LTR)
    #plt.plot_histogram(DP.T, LTR)

    # plt.plot_correlations(DTE.T, "heatmap")
    # plt.plot_correlations(DTE.T[:, LTE == 0], "heatmap_spoofed_", cmap="Reds")
    # plt.plot_correlations(DTE.T[:, LTE == 1], "heatmap_authentic_", cmap="Blues")

    validation(DTR,LTR,VA,dimRed)
    evaluation(DTE, LTE, DTR, LTR, EV, dimRed)

    #plt.plot_correlations(DTR.T,"heatmap")
    #plt.plot_correlations(DTR.T[:, LTR == 0], "heatmap_spoofed_", cmap="Reds")
    #plt.plot_correlations(DTR.T[:, LTR == 1], "heatmap_authentic_", cmap="Blues")



