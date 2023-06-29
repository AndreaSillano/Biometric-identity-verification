

import matplotlib.pyplot as plt
import numpy

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
def plotPCAScatter(D,L):

    Dx0 = (D[0, L == 0])
    Dy0 = (D[1, L == 0])
    plt.scatter(Dx0, Dy0, color="#E23A2E", label="Different Speaker")
    Dx1 = (D[0, L == 1])
    Dy1 = (D[1, L == 1])
    plt.scatter(Dx1, Dy1, color="#279847", label="Same Speaker")
    plt.legend(loc='upper right')


    plt.show()

def PCA(D,m,C):
    U, s, Vh = numpy.linalg.svd(C)

    P = U[:, 0:m]
    #P = numpy.dot(P, [[1, 0], [0, -1]])
    DP = numpy.dot(P.T, D)
    return DP

def wrapperPCA(D,LTR):
    D = D.transpose()
    mu = empirical_mean(D)
    #DC = D - D.mean(1).reshape((D.shape[0], 1))
    #C = numpy.dot(DC, DC.T)
    #C = C / float(DC.shape[1])
    C = empirical_covariance(D, mu)

    DP = PCA(D, 2, C)
    print(DP)
    plotPCAScatter(DP,LTR)
def computeSW(D, L):
    D0 = D[:, L==0 ]
    D1 = D[:, L == 1]

    DC0  = D0 - D0.mean(1).reshape((D0.shape[0], 1))
    DC1 = D1 - D1.mean(1).reshape((D1.shape[0], 1))

    C0 = numpy.dot(DC0,DC0.T)
    C1 = numpy.dot(DC1, DC1.T)
    return (C0+C1)/float(D.shape[1])

def computeSB(D,L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    mu = empirical_mean(D)
    Dmu0 = empirical_mean(D0) -mu
    Dmu1 = empirical_mean(D1) - mu
    CM0 = numpy.outer(Dmu0,Dmu0)*D0.shape[1]
    CM1 = numpy.outer(Dmu1, Dmu1) * D1.shape[1]

    return (CM0+CM1)/float(D.shape[1])

def LDAByJointDiag(SB,SW, m, D):
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot( numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T )
    SBT = numpy.dot(P1,SB)
    SBT = numpy.dot(SBT,P1.T)

    U, s, Vh = numpy.linalg.svd(SBT)

    P2 = U[:, 0:m]
    P2 = numpy.dot(P2,[[-1,0],[0,-1]])
    W = numpy.dot(P1.T, P2)
    DP = numpy.dot(W.T, D)

    return DP
def wrapperLDA(D, LTR):
    D = D.transpose()
    SW = computeSW(D, LTR)
    SB = computeSB(D,LTR)
    DP = LDAByJointDiag(SB,SW,2,D)
    plotPCAScatter(DP,LTR)

if __name__ == "__main__":
    #Load Data & Test Set + Randomize
    DTR,LTR = load("Train.txt")
    #DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    #DTE, LTE = randomize(DTE, LTE)
    #Plot
    #plotHist(DTR, LTR)
    #plotScatter(DTR,LTR)
    #PCA
    print("PRINCIPAL COMPONENT ANALYSIS")
    wrapperPCA(DTR,LTR)
    print("LINEAR DISCRIMINANT ANALYSIS")
    wrapperLDA(DTR,LTR)

    