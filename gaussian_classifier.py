import numpy
from mlFunc import *
class MultivariateGaussianClassifier:

    def __init__(self):
        self.classes = None
        self.means = []
        self.covariances = []

    def setup_MVG(self,D, L):
        self.classes = numpy.unique(L)
        for c in self.classes:
            D_c = D[:, L == c]
            mu = empirical_mean(D_c)
            self.means.append(mu)
            C = empirical_covariance(D_c, mu)
            self.covariances.append(C)

    def predict_MVG(self, D, L):
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
        print("LOG predicion Post Probability")
        print("ACCURACY: ", acc, "%")
        err = 100 - (acc)
        print("ERROR: ", err, "%")


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





