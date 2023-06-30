

import matplotlib.pyplot as plt
import numpy
from plotter import Plotter
from dimensionality_reduction import DimensionalityReduction
from gaussian_classifier import MultivariateGaussianClassifier
from logistic_regression import LogisticRegression
from mlFunc import *
def plot_histograms(embeddings, labels):
    num_dimensions = embeddings.shape[1]

    for dim in range(num_dimensions):
        fig, ax = plt.subplots()

        # Separate embeddings based on labels
        same_speaker_embeddings = embeddings[labels == 1, dim]
        diff_speaker_embeddings = embeddings[labels == 0, dim]

        # Plot histograms for same and different speaker embeddings
        ax.hist(same_speaker_embeddings, bins=10, alpha=0.5, label='Authentic')
        ax.hist(diff_speaker_embeddings, bins=10, alpha=0.5, label='Spoofed')

        ax.set_xlabel(f"Dimension {dim+1}")
        ax.set_ylabel("Frequency")
        ax.legend()

        plt.show()

def plot_scatter(embeddings, labels):
    num_dimensions = embeddings.shape[1]

    for dim1 in range(num_dimensions - 1):
        for dim2 in range(dim1 + 1, num_dimensions):
            fig, ax = plt.subplots()

            # Separate embeddings based on labels
            same_speaker_embeddings = embeddings[labels == 1]
            diff_speaker_embeddings = embeddings[labels == 0]

            # Plot scatter plot for same and different speaker embeddings
            ax.scatter(same_speaker_embeddings[:, dim1], same_speaker_embeddings[:, dim2], label='Same Speaker')
            ax.scatter(diff_speaker_embeddings[:, dim1], diff_speaker_embeddings[:, dim2], label='Different Speaker')

            ax.set_xlabel(f"Dimension {dim1 + 1}")
            ax.set_ylabel(f"Dimension {dim2 + 1}")
            ax.legend()

            plt.show()



if __name__ == "__main__":

    DTR,LTR = load("Train.txt")

    DTE, LTE = load("Test.txt")

    plt = Plotter()
    dimRed = DimensionalityReduction()
    MVG = MultivariateGaussianClassifier()
    LR = LogisticRegression()

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

    plot_correlations(DTR,"heatmap")
    plot_correlations(DTR.T[:, LTR == 0], "heatmap_spoofed_", cmap="Reds")
    plot_correlations(DTR.T[:, LTR == 1], "heatmap_authentic_", cmap="Blues")


    print("---------------MVG WITHOUT LDA--------------------------")

    MVG.setup_MVG(DTR.T,LTR)
    MVG.predict_MVG(DTE.T,LTE)

    print("---------------MVG WITH LDA--------------------------")
    MVG.setup_MVG(DP,LTR)
    MVG.predict_MVG(DPE,LTE)

    print("---------------MVG NAIVE BAYES WITHOUT LDA--------------------------")
    MVG.setup_MVG_Naive_Bayes(DTR.T, LTR)
    MVG.predict_MVG_Naive_Bayes(DTE.T, LTE)

    print("---------------MVG NAIVE BAYES WITH LDA--------------------------")

    MVG.setup_MVG_Naive_Bayes(DP, LTR)
    MVG.predict_MVG_Naive_Bayes(DPE, LTE)

    print("---------------MVG TIED COV WITHOUT LDA--------------------------")

    MVG.setup_MVG_Tied_Cov(DTR.T, LTR)
    MVG.predict_MVG_Tied_Cov(DTE.T, LTE)

    print("---------------MVG TIED COV WITH LDA--------------------------")

    MVG.setup_MVG_Tied_Cov(DP, LTR)
    MVG.predict_MVG_Tied_Cov(DPE, LTE)

    print("---------------MVG TIED COV + NAIVE WITHOUT LDA--------------------------")

    MVG.setup_MVG_Tied_Cov_Naive(DTR.T, LTR)
    MVG.predict_MVG_Tied_Cov_Naive(DTE.T, LTE)

    print("---------------MVG TIED COV + NAIVE WITH LDA--------------------------")
    MVG.setup_MVG_Tied_Cov_Naive(DP, LTR)
    MVG.predict_MVG_Tied_Cov_Naive(DPE, LTE)

    print("---------------LOGISTIC REGRESSION WITHOUT LDA--------------------------")
    LR.setup_Logistic_Regression(DTR.T,LTR, 0.1)
    LR.preditc_Logistic_Regression(DTE.T,LTE,0.1)

    print("---------------LOGISTIC REGRESSION WITH LDA--------------------------")
    LR.setup_Logistic_Regression(DP,LTR, 0.1)
    LR.preditc_Logistic_Regression(DPE,LTE,0.1)