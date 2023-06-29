from mlFunc import *

def plotHist(D, L):
    D = D.transpose()
    D0 = sorted(D[0, L==0])
    D1 = sorted(D[0, L==1])

    plt.hist(D0, bins = 100, density=True, ec='black', color="#E23A2E", alpha = 0.5, label="Different Speaker")
    plt.hist(D1, bins = 100, density=True, ec='black', color="#279847", alpha = 0.5, label="Same Speaker")

    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    #Load Data & Test Set + Randomize
    DTR,LTR = load("Train.txt")
    #DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    DTE, LTE = randomize(DTE, LTE)
    plotHist(DTR, LTR)
    