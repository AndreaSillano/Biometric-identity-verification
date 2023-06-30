

import matplotlib.pyplot as plt
import numpy
from plotter import Plotter
from dimensionality_reduction import DimensionalityReduction
from mlFunc import *
def plot_histograms(embeddings, labels):
    num_dimensions = embeddings.shape[1]

    for dim in range(num_dimensions):
        fig, ax = plt.subplots()

        # Separate embeddings based on labels
        same_speaker_embeddings = embeddings[labels == 1, dim]
        diff_speaker_embeddings = embeddings[labels == 0, dim]

        # Plot histograms for same and different speaker embeddings
        ax.hist(same_speaker_embeddings, bins=10, alpha=0.5, label='Same Speaker')
        ax.hist(diff_speaker_embeddings, bins=10, alpha=0.5, label='Different Speaker')

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


    #plt.plot_histogram(DTR, LTR)
    #plt.plot_scatter(DTR, LTR)

    print("---------------PRINCIPAL COMPONENT ANALYSIS-------------")
    DP = dimRed.PCA(DTR, 2)
    plt.plot_PCA_scatter(DP,LTR)
    dimRed.evaluatePCA(DP,LTR)
    print("---------------LINEAR DISCRIMINANT ANALYSIS-------------")
    DP = dimRed.LDA(DTR,LTR)
    plt.plot_LDA_scatter(DP,LTR)

