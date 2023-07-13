import numpy.linalg

from mlFunc import *


class DimensionalityReduction:
    # ----------PRINCIPAL COMPONENT ANALISYS---------------
    def PCA(self,D, m):
        D = D.transpose()

        DC = D - D.mean(1).reshape((D.shape[0], 1))
        C = numpy.dot(DC, DC.T)
        C = C / float(DC.shape[1])

        DP = self._computePCA(D, m, C)
        return DP

    def PCA_DTE(self,D, m, DTE):
        D = D.transpose()
        DTE = DTE.transpose()
        DC = D - D.mean(1).reshape((D.shape[0], 1))
        C = numpy.dot(DC, DC.T)
        C = C / float(DC.shape[1])

        DPE =  self._computePCA(DTE, m, C)
        return DPE

    def _computePCA(self, D, m, C):
        U, s, Vh = numpy.linalg.svd(C)
        P = U[:, 0:m]
        #P = numpy.dot(P, [[1, 0], [0, -1]])
        DP = numpy.dot(P.T, D)
        return DP



    def evaluatePCA(self, DP, L):
        D0 = DP[:, L == 0]
        D1 = DP[:, L == 1]
        mu0 = empirical_mean(D0)
        mu1 = empirical_mean(D1)
        means_c = [mu0,mu1]
        pred = []
        for sample in DP.T:
            distances = [numpy.linalg.norm(sample-mu_c) for mu_c in means_c ]
            closest_class = numpy.argmin(distances)
            pred.append(closest_class)

        correct_predictions = 0
        total_samples = len(L)

        for true_label, pred_label in zip(L, pred):
            if true_label == pred_label:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples

        # Print accuracy
        print("Accuracy:", accuracy*100)

    #----------LINEAR DISCRIMINANT ANALISYS---------------
    def LDA(self,D,LTR):
        D = D.transpose()
        SW = self._computeSW(D, LTR)
        SB = self._computeSB(D, LTR)
        DP = self._LDAByJointDiag(SB, SW, 2, D)
        return DP

    def _computeSW(self,D, L):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]

        DC0 = D0 - D0.mean(1).reshape((D0.shape[0], 1))
        DC1 = D1 - D1.mean(1).reshape((D1.shape[0], 1))

        C0 = numpy.dot(DC0, DC0.T)
        C1 = numpy.dot(DC1, DC1.T)
        return (C0 + C1) / float(D.shape[1])

    def _computeSB(self,D, L):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        mu = empirical_mean(D)
        Dmu0 = empirical_mean(D0) - mu
        Dmu1 = empirical_mean(D1) - mu
        CM0 = numpy.outer(Dmu0, Dmu0) * D0.shape[1]
        CM1 = numpy.outer(Dmu1, Dmu1) * D1.shape[1]

        return (CM0 + CM1) / float(D.shape[1])

    def _LDAByJointDiag(self,SB, SW, m, D):
        U, s, _ = numpy.linalg.svd(SW)
        P1 = numpy.dot(numpy.dot(U, numpy.diag(1.0 / (s ** 0.5))), U.T)
        SBT = numpy.dot(P1, SB)
        SBT = numpy.dot(SBT, P1.T)

        U, s, Vh = numpy.linalg.svd(SBT)

        P2 = U[:, 0:m]
        P2 = numpy.dot(P2, [[-1, 0], [0, -1]])
        W = numpy.dot(P1.T, P2)
        DP = numpy.dot(W.T, D)

        return DP