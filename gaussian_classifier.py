import numpy
from mlFunc import *
import scipy
class MultivariateGaussianClassifier:



    def predict_MVG(self, D, L, DTE):

        classes = numpy.unique(L)
        covariances =[]
        means = []
        logSJoint = numpy.zeros((2, DTE.shape[1]))
        dens = numpy.zeros((2, DTE.shape[1]))

        for c in classes:
            D_c = D[:, L == c]
            mu = empirical_mean(D_c)
            means.append(mu)
            C = empirical_covariance(D_c, mu)
            covariances.append(C)

        for label in classes:
            dens[label, :] = numpy.exp(self.logpdf_GAU_ND(DTE, means[label], covariances[label]).ravel())
            logSJoint[label, :] = self.logpdf_GAU_ND(DTE, means[label], covariances[label]).ravel() + numpy.log(1/2)

        logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)
        logPost = logSJoint - vrow(logSMarginal)
        Post = numpy.exp(logPost)
        pred = numpy.argmax(Post, axis=0)

        # common_elements = []
        # for i in range(len(pred)):
        #     if pred[i] == L[i]:
        #         common_elements.append(L[i])
        #
        # acc = len(common_elements) / len(L) * 100
        # print("LOG predicion Post Probability")
        # print("ACCURACY: ", acc, "%")
        # err = 100 - (acc)
        # print("ERROR: ", err, "%")
        return numpy.log(dens[1] / dens[0])



    def logpdf_GAU_ND(self, X, mu, C):
        Y = []
        for i in range(X.shape[1]):
            Y.append(self.logpdf_GAU_ND1(X[:, i:i + 1], mu, C))
        return numpy.array(Y).ravel()

    def logpdf_GAU_ND1(self, x, mu, C):
        xc = x - mu
        M = x.shape[0]
        const = -0.5 * M * numpy.log(2 * numpy.pi)
        logdet = numpy.linalg.slogdet(C)[1]
        L = numpy.linalg.inv(C)
        v = numpy.dot(xc.T, numpy.dot(L, xc)).ravel()
        return const - 0.5 * logdet - 0.5 * v

    #NAIVE BAYES


    def predict_MVG_Naive_Bayes(self, D, L,DTE):

        classes = numpy.unique(L)
        covariances = []
        means = []
        logSJoint = numpy.zeros((2, DTE.shape[1]))
        dens = numpy.zeros((2, DTE.shape[1]))

        for c in classes:
            D_c = D[:, L == c]
            mu = empirical_mean(D_c)
            means.append(mu)
            C = computeCovDiag(D_c, mu)
            covariances.append(C)

        for label in classes:
            dens[label, :] = numpy.exp(self.logpdf_GAU_ND(DTE, means[label], covariances[label]).ravel())
            logSJoint[label, :] = self.logpdf_GAU_ND(DTE, means[label], covariances[label]).ravel() + numpy.log(1 / 2)

        logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)
        logPost = logSJoint - vrow(logSMarginal)
        Post = numpy.exp(logPost)
        pred = numpy.argmax(Post, axis=0)

        # common_elements = []
        # for i in range(len(pred)):
        #     if pred[i] == L[i]:
        #         common_elements.append(L[i])
        #
        # acc = len(common_elements) / len(L) * 100
        # print("LOG predicion Post Probability")
        # print("ACCURACY: ", acc, "%")
        # err = 100 - (acc)
        # print("ERROR: ", err, "%")
        return numpy.log(dens[1] / dens[0])

    #MVG TIED COV

    def _computeSW(self, D, L):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]

        DC0 = D0 - D0.mean(1).reshape((D0.shape[0], 1))
        DC1 = D1 - D1.mean(1).reshape((D1.shape[0], 1))

        C0 = numpy.dot(DC0, DC0.T)
        C1 = numpy.dot(DC1, DC1.T)
        return (C0 + C1) / float(D.shape[1])

    def setup_MVG_Tied_Cov(self, D, L):
        self.means = []
        self.covariances = []
        self.classes = numpy.unique(L)
        for c in self.classes:
            D_c = D[:, L == c]
            mu = empirical_mean(D_c)
            self.means.append(mu)
            C = self._computeSW(D, L)
            self.covariances.append(C)

    def predict_MVG_Tied_Cov(self, D, L):
        ll = []
        for mu, C in zip(self.classes, self.covariances):
            ll.append(self.logpdf_GAU_ND(D, mu, C))

        Sjoint = numpy.exp(ll)*(1/2)
        Smarginal = vrow(Sjoint.sum(0))
        SPost = Sjoint/Smarginal
        pred = numpy.argmax(SPost, axis=0)

        common_elements = []
        for i in range(len(pred)):
            if pred[i] == L[i]:
                common_elements.append(L[i])

        acc = len(common_elements) / len(L) * 100
        print("MVG TIED COV predicion Post Probability")
        print("ACCURACY: ", acc, "%")
        err = 100 - (acc)
        print("ERROR: ", err, "%")
        return (numpy.log(numpy.exp(ll[0])/numpy.exp(ll[1])))
    #BYES + TIED
    def _computeSW_Diag(self, D, L):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]

        DC0 = D0 - D0.mean(1).reshape((D0.shape[0], 1))
        DC1 = D1 - D1.mean(1).reshape((D1.shape[0], 1))

        C0 = numpy.dot(DC0, DC0.T)
        C1 =  numpy.dot(DC1, DC1.T)

        return numpy.diag(numpy.diag((C0 + C1) / float(D.shape[1])))

    def setup_MVG_Tied_Cov_Naive(self, D, L):
        self.means = []
        self.covariances = []
        self.classes = numpy.unique(L)
        for c in self.classes:
            D_c = D[:, L == c]
            mu = empirical_mean(D_c)
            self.means.append(mu)
            C = self._computeSW_Diag(D, L)
            self.covariances.append(C)

    def predict_MVG_Tied_Cov_Naive(self, D, L):
        ll = []
        for mu, C in zip(self.classes, self.covariances):
            ll.append(self.logpdf_GAU_ND(D, mu, C))

        Sjoint = numpy.exp(ll)*(1/2)
        Smarginal = vrow(Sjoint.sum(0))
        SPost = Sjoint/Smarginal
        pred = numpy.argmax(SPost, axis=0)

        common_elements = []
        for i in range(len(pred)):
            if pred[i] == L[i]:
                common_elements.append(L[i])

        acc = len(common_elements) / len(L) * 100
        print("MVG TIED COV + Bayes predicion Post Probability")
        print("ACCURACY: ", acc, "%")
        err = 100 - (acc)
        print("ERROR: ", err, "%")
        return (numpy.log(numpy.exp(ll[1])/numpy.exp(ll[0])))


