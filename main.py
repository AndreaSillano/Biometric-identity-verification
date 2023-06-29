from mlFunc import *

def plotHist(D, L):
    D0 = sorted(D[0, L==0])
    D1 = sorted(D[0, L==1])
    print(D)
    plt.hist(D0, bins = 20, density=True, ec='black', color="#F77C3F", alpha = 0.5, label="Different Speaker")
    plt.hist(D1, bins = 20, density=True, ec='black', color="#F77C3F", alpha = 0.5, label="Same Speaker")
    plt.show()

if __name__ == "__main__":

    DTR,LTR = load("Train.txt")
    DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    DTE, LTE = randomize(DTE, LTE)
    plotHist(DTR, LTR)
    