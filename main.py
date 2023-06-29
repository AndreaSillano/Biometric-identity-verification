import matplotlib.pyplot as plt

from mlFunc import *

def plotHist(D, L):

    D = D.transpose()

    for i in range(0,D.shape[0]):
        D0 = (D[:, L==0])[i]
        D1 = (D[:, L==1])[i]

        plt.hist(D0, bins = 100, density=True, ec='black', color="#E23A2E", alpha = 0.5, label="Different Speaker")
        plt.hist(D1, bins = 100, density=True, ec='black', color="#279847", alpha = 0.5, label="Same Speaker")

        plt.legend(loc='upper right')
        plt.savefig("./images/hist/hist_" + str(i) + ".png")
        plt.show()



if __name__ == "__main__":
    #Load Data & Test Set + Randomize
    DTR,LTR = load("Train.txt")
    #DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    #DTE, LTE = randomize(DTE, LTE)
    plotHist(DTR, LTR)
    