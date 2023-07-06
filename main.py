

import matplotlib.pyplot as plt
import numpy
from plotter import Plotter
from dimensionality_reduction import DimensionalityReduction
from validator import Validation
from mlFunc import *



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

    #plt.plot_histogram(DTE,LTE)
    #plt.plot_histogram(DTR, LTR)
    #plt.plot_scatter(DTR, LTR)

    print("---------------PRINCIPAL COMPONENT ANALYSIS-------------")
    DPA = dimRed.PCA(DTR, 8)
    DPEA = dimRed.PCA(DTR, 2)
    plt.plot_PCA_scatter(DPA,LTR)
    dimRed.evaluatePCA(DPA,LTR)
    print("---------------LINEAR DISCRIMINANT ANALYSIS-------------")
    DP = dimRed.LDA(DTR,LTR)
    DPE = dimRed.LDA(DTE,LTE)
    plt.plot_LDA_scatter(DP,LTR)

    #plt.plot_correlations(DTR.T,"heatmap")
    #plt.plot_correlations(DTR.T[:, LTR == 0], "heatmap_spoofed_", cmap="Reds")
    #plt.plot_correlations(DTR.T[:, LTR == 1], "heatmap_authentic_", cmap="Blues")


    # VA.MVG_validation(DTR,LTR, 0.9, 1,1,DTE,LTE)
    # for i in range (7,10):
    #     print("PCA con", i)
    #     DPA = dimRed.PCA(DTR, i)
    #     VA.MVG_validation(DPA.T, LTR, 0.9, 1, 1, DTE, LTE)


    #VA.LR_validation(DTR,LTR, 0.5,1,10)

    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    for k in K_arr:
        for c in C_arr:
            print("SVM, K: ",k," C: ", c)
            VA.SVM_validation(DTR, LTR, 0.5, 1, 10, k, c)
    #VA.GMM_validation(DTR,LTR, 0.5,1,10, 2,8, 0.1, 0.01)
