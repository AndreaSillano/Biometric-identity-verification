import numpy
import scipy
from scipy.optimize import fmin_l_bfgs_b
from mlFunc import *

class SupportVectorMachine:
    def __init__(self):
        self.w = []
        self.b = []
        self.primal_loss = []
        self.dual_loss = []
        self.duality_gap = []

    def setup_primal_svm(self, D, L, C):
        n_samples = D.shape[0]

        X_extended = numpy.hstack((D, numpy.ones((n_samples, 1))))

        Gb = numpy.dot(X_extended, X_extended.T)
        Hc = Gb * L[:, numpy.newaxis] * L

        def objective_function(alpha):
            H_alpha_c = numpy.dot(Hc, alpha)
            #LbD = 0.5 * numpy.dot(alpha.T, H_alpha_c) - numpy.dot(alpha.T, numpy.ones(n_samples))
            LbD = 0.5 * numpy.dot(alpha, H_alpha_c) - numpy.sum(alpha)
            gradient = H_alpha_c - 1.0
            return LbD, gradient

        bounds = [(0, C)] * n_samples

        alpha_init = numpy.zeros(n_samples)
        alpha_star, _, _ = fmin_l_bfgs_b(objective_function, alpha_init, bounds=bounds)
        
        wb_star = numpy.sum((numpy.dot(alpha_star * L[:, numpy.newaxis], X_extended)), axis=0)
        self.w = wb_star[:-1]
        self.b = wb_star[-1]

        self.primal_loss = 0.5 * numpy.dot(self.w, self.w) + C * numpy.sum(numpy.maximum(0, 1 - L * numpy.dot(self.w, D.T + self.b)))
        self.dual_loss = 0.5 * numpy.dot(alpha_star.T, numpy.dot(Hc, alpha_star)) - numpy.sum(alpha_star)
        self.duality_gap = self.primal_loss - self.dual_loss


    def predict_primal_svm(self, DTE, LTE, C):
        s = numpy.dot(self.w.T, DTE) + self.b
        print(s)
        my_pred = []
        correct = 0
        for p in s:
            if p > 0:
                my_pred.append(1)
            else:
                my_pred.append(0)

        for i in range(0, len(LTE)):
            if LTE[i] == my_pred[i]:
                correct += 1

        accuracy = correct / len(LTE)
        err = (1 - accuracy)*100

        # print(my_pred)
        # print(LTE)
        print("ACCURACY: ", accuracy*100, "C: ", C)
        print("ERROR: ", err, "%")
        print("Primal loss: ",self.primal_loss, " Dual loss: ",self.dual_loss, " Duality gap: ",self.duality_gap)
