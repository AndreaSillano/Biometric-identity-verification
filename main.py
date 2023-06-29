

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

def plotScatter(D,L):
    D = D.transpose()

    for i in range(D.shape[0]):
        for j in range(0, D.shape[0]):
            if j != i:
                Dx0 = (D[i, L == 0])
                Dy0 = (D[j, L == 0])
                plt.scatter(Dx0,Dy0, color="#E23A2E", label="Different Speaker")
                Dx1 = (D[i, L == 1])
                Dy1 = (D[j, L == 1])
                plt.scatter(Dx1, Dy1, color="#279847", label="Same Speaker")
                plt.legend(loc='upper right')
                plt.savefig("./images/scatter/scatter"+str(i)+"_"+str(j)+".png")

                plt.close()

                #plt.show()


if __name__ == "__main__":
    #Load Data & Test Set + Randomize
    DTR,LTR = load("Train.txt")
    #DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    #DTE, LTE = randomize(DTE, LTE)
    #plotHist(DTR, LTR)
    plotScatter(DTR,LTR)
    