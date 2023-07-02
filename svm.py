import numpy
from scipy.optimize import fmin_l_bfgs_b
from mlFunc import *
from itertools import repeat

class SupportVectorMachine:
    def __init__(self):
        self.w = []
        self.pl = []
        self.dl = []
        self.dg = []

    def setup_primal_svm(self, DTR, LTR, C, K=1):
        row = numpy.zeros(DTR.shape[1])+K
        D = numpy.vstack([DTR, row])
        
        # Compute the H matrix exploiting broadcasting
        Gij = numpy.dot(D.T, D)
        # To compute zi*zj I need to reshape LTR as a matrix with one column/row
        # and then do the dot product
        zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
        Hij = zizj*Gij

        def objective_function(alpha):
            #H_alpha_c = numpy.dot(Hij, alpha)
            #LbD = 0.5 * numpy.dot(alpha.T, H_alpha_c) - numpy.dot(alpha.T, numpy.ones(n_samples))
            #LbD = 0.5*numpy.dot(numpy.dot(alpha.T, Hij), alpha)-numpy.dot(alpha.T, numpy.ones(Hij.shape[1]))
            #gradient = H_alpha_c - numpy.ones(Hij.shape[1])
            #return LbD, gradient
            grad = numpy.dot(Hij, alpha) - numpy.ones(Hij.shape[1])
            return ((1/2)*numpy.dot(numpy.dot(alpha.T, Hij), alpha)-numpy.dot(alpha.T, numpy.ones(Hij.shape[1])), grad)

        bounds = list(repeat((0, C), DTR.shape[1]))

        alpha_init = numpy.zeros(DTR.shape[1])
        x, f, d = fmin_l_bfgs_b(objective_function, alpha_init, bounds=bounds)
        
        wb_star = numpy.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)
        self.w = wb_star#[:-1]
        #self.b = wb_star[-1]
        self.pl, self.dl, self.dg = self.primalObjective(wb_star, D, C, LTR, f)

    def primalObjective(self, w, D, C, LTR, f):
        normTerm = (1/2)*(numpy.linalg.norm(w)**2)
        m = numpy.zeros(LTR.size)
        for i in range(LTR.size):
            vett = [0, 1-LTR[i]*(numpy.dot(w.T, D[:, i]))]
            m[i] = vett[numpy.argmax(vett)]
        pl = normTerm + C*numpy.sum(m)
        dl = -f
        dg = pl-dl
        return pl, dl, dg

    def predict_primal_svm(self, DTE, LTE, C, K=1):
        row = numpy.zeros(DTE.shape[1])+K
        DTE = numpy.vstack([DTE, row])
        #print("LTE: ", LTE)
        #S = numpy.dot(self.w.T, DTE) + self.b
        S = numpy.dot(self.w.T, DTE)

        #print(S)
        my_pred = []
        correct = 0
        for p in S:
            if p > 0:
                my_pred.append(1)
            else:
                my_pred.append(0)

        for i in range(0, len(LTE)):
            if LTE[i] == my_pred[i]:
                correct += 1

        accuracy = correct / len(LTE)
        err = (1 - accuracy)*100

        print("ACCURACY: ", accuracy*100, "C: ", C)
        print("ERROR: ", err, "%")
        print("K=%d, C=%f, Primal loss=%e, Dual loss=%e, Duality gap=%e, Error rate=%.1f %%" % (K, C, self.pl, self.dl, self.dg, err))
    
    def dualLossErrorRatePoly(self, DTR, C, Hij, LTR, LTE, DTE, K, d, c):
        def objective_function(alpha):
            #H_alpha_c = numpy.dot(Hij, alpha)
            #LbD = 0.5 * numpy.dot(alpha.T, H_alpha_c) - numpy.dot(alpha.T, numpy.ones(n_samples))
            #LbD = 0.5*numpy.dot(numpy.dot(alpha.T, Hij), alpha)-numpy.dot(alpha.T, numpy.ones(Hij.shape[1]))
            #gradient = H_alpha_c - numpy.ones(Hij.shape[1])
            #return LbD, gradient
            grad = numpy.dot(Hij, alpha) - numpy.ones(Hij.shape[1])
            return ((1/2)*numpy.dot(numpy.dot(alpha.T, Hij), alpha)-numpy.dot(alpha.T, numpy.ones(Hij.shape[1])), grad)
        
        b = list(repeat((0, C), DTR.shape[1]))
        (x, f, data) = fmin_l_bfgs_b(objective_function,
                                        numpy.zeros(DTR.shape[1]), bounds=b)
        # Compute the scores
        S = numpy.sum(
            numpy.dot((x*LTR).reshape(1, DTR.shape[1]), (numpy.dot(DTR.T, DTE)+c)**d+ K), axis=0)
        # Compute predicted labels. 1* is useful to convert True/False to 1/0
        LP = 1*(S > 0)
        # Replace 0 with -1 because of the transformation that we did on the labels
        LP[LP == 0] = -1
        numberOfCorrectPredictions = numpy.array(LP == LTE).sum()
        accuracy = numberOfCorrectPredictions/LTE.size*100
        errorRate = 100-accuracy
        # Compute dual loss
        dl = -f
        print("K=%d, C=%f, Kernel Poly (d=%d, c=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, d, c, dl, errorRate))
        return

    def setup_kernelPoly_svm(self, DTR, LTR, DTE, LTE, K=1, C=1, d=2, c=0):
        # Compute the H matrix exploiting broadcasting
        kernelFunction = (numpy.dot(DTR.T, DTR)+c)**d+ K**2
        # To compute zi*zj I need to reshape LTR as a matrix with one column/row
        # and then do the dot product
        zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
        Hij = zizj*kernelFunction
        # We want to maximize JD(alpha), but we can't use the same algorithm of the
        # previous lab, so we can cast the problem as minimization of LD(alpha) defined
        # as -JD(alpha)
        self.dualLossErrorRatePoly(DTR, C, Hij, LTR, LTE, DTE, K, d, c)
        return

    def dualLossErrorRateRBF(self, DTR, C, Hij, LTR, LTE, DTE, K, gamma):
        def objective_function(alpha):
            H_alpha_c = numpy.dot(Hij, alpha)
            #LbD = 0.5 * numpy.dot(alpha.T, H_alpha_c) - numpy.dot(alpha.T, numpy.ones(n_samples))
            LbD = 0.5*numpy.dot(numpy.dot(alpha.T, Hij), alpha)-numpy.dot(alpha.T, numpy.ones(Hij.shape[1]))
            gradient = H_alpha_c - numpy.ones(Hij.shape[1])
            return LbD, gradient
        
        b = list(repeat((0, C), DTR.shape[1]))
        (x, f, data) = fmin_l_bfgs_b(objective_function,
                                        numpy.zeros(DTR.shape[1]), bounds=b)
        kernelFunction = numpy.zeros((DTR.shape[1], DTE.shape[1]))
        for i in range(DTR.shape[1]):
            for j in range(DTE.shape[1]):
                kernelFunction[i,j]=self.RBF(DTR[:, i], DTE[:, j], gamma, K)
        S=numpy.sum(numpy.dot((x*LTR).reshape(1, DTR.shape[1]), kernelFunction), axis=0)
        # Compute the scores
        # S = np.sum(
        #     np.dot((x*LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DTE)+c)**d+ K), axis=0)
        # Compute predicted labels. 1* is useful to convert True/False to 1/0
        LP = 1*(S > 0)
        # Replace 0 with -1 because of the transformation that we did on the labels
        LP[LP == 0] = -1
        numberOfCorrectPredictions = numpy.array(LP == LTE).sum()
        accuracy = numberOfCorrectPredictions/LTE.size*100
        errorRate = 100-accuracy
        # Compute dual loss
        dl = -f
        print("K=%d, C=%f, RBF (gamma=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, gamma, dl, errorRate))
        return
    
    def RBF(self, x1, x2, gamma, K):
        #print(numpy.exp(-gamma*(numpy.linalg.norm(x1-x2)**2))+K**2)
        return numpy.exp(-gamma * numpy.linalg.norm(x1 - x2, axis=1) ** 2) + K ** 2
        #return numpy.exp(-gamma*(numpy.linalg.norm(x1-x2)**2))+K**2
        

    def setup_kernelRBF_svm(self, DTR, LTR, DTE, LTE, K=0, C=1, gamma=1):
        # Compute the H matrix exploiting broadcasting
        kernelFunction = numpy.zeros((DTR.shape[1], DTR.shape[1]))
        #for i in range(DTR.shape[1]):
            #for j in range(DTR.shape[1]):
                #kernelFunction[i,j]=self.RBF(DTR[:, i], DTR[:, j], gamma, K)
        X1, X2 = numpy.meshgrid(DTR.T, DTR.T)
        kernelFunction = self.RBF(X1.T, X2.T, gamma, K)
  
        # To compute zi*zj I need to reshape LTR as a matrix with one column/row
        # and then do the dot product
        zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
        Hij = zizj*kernelFunction
        # We want to maximize JD(alpha), but we can't use the same algorithm of the
        # previous lab, so we can cast the problem as minimization of LD(alpha) defined
        # as -JD(alpha)
        self.dualLossErrorRateRBF(DTR, C, Hij, LTR, LTE, DTE, K, gamma)
        return