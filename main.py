

import matplotlib.pyplot as plt
import numpy
from plotter import Plotter
from dimensionality_reduction import DimensionalityReduction
from validator import  Validation
from gaussian_classifier import MultivariateGaussianClassifier
from logistic_regression import LogisticRegression
from mlFunc import *



if __name__ == "__main__":

    DTR,LTR = load("Train.txt")

    DTE, LTE = load("Test.txt")

    #DTR_Z= DTR- DTR.mean(1).reshape((DTR.shape[0], 1))
    #DTE_Z= DTE- DTR.mean(1).reshape((DTR.shape[0], 1))

    print("TRAINING AUTHENTIC: ", DTR.T[:,LTR==1].shape[1])
    print("TRAINING SPOFFED", DTR.T[:, LTR==0].shape[1])

    print("TEST AUTHENTIC: ", DTE.T[:, LTE == 1].shape[1])
    print("TEST SPOFFED", DTE.T[:, LTE == 0].shape[1])

    plt = Plotter()
    dimRed = DimensionalityReduction()
    MVG = MultivariateGaussianClassifier()
    LR = LogisticRegression()
    
    VA = Validation()
    #plt.plot_histogram(DTE,LTE)
    #plt.plot_histogram(DTR, LTR)
    #plt.plot_scatter(DTR, LTR)

    print("---------------PRINCIPAL COMPONENT ANALYSIS-------------")
    DPA = dimRed.PCA(DTR, 2)
    DPEA = dimRed.PCA(DTR, 2)
    plt.plot_PCA_scatter(DPA,LTR)
    dimRed.evaluatePCA(DPA,LTR)
    print("---------------LINEAR DISCRIMINANT ANALYSIS-------------")
    DP = dimRed.LDA(DTR,LTR)
    DPE = dimRed.LDA(DTE,LTE)
    plt.plot_LDA_scatter(DP,LTR)

    plot_correlations(DTR.T,"heatmap")
    plot_correlations(DTR.T[:, LTR == 0], "heatmap_spoofed_", cmap="Reds")
    plot_correlations(DTR.T[:, LTR == 1], "heatmap_authentic_", cmap="Blues")


    VA.MVG_validation(DTR,LTR,DTE,LTE)
