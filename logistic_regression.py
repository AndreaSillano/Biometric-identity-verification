import numpy
import scipy
from scipy import optimize
from mlFunc import *

class LogisticRegression:

    def logreg_obj(self,v, DTR, LTR, l):
        w, b = v[0:-1], v[-1]
        zi = 2 * LTR - 1
        f = (l / 2) * numpy.sum(w ** 2) + (1 / DTR.shape[1]) * numpy.sum(
            numpy.logaddexp(0, -zi * (numpy.dot(w.T, DTR) + b)))
        return (f)

    def predict_Logistic_Regression_weigthed(self,D,L,DTE,l,pi):
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj_weighted, numpy.zeros(D.shape[0] + 1),
                                                  approx_grad=True, args=(D, L, l, pi))
        _w = _v[0:D.shape[0]]
        _b = _v[-1]
        s = numpy.dot(numpy.array(_w).T, DTE) + _b
        return s
    def preditc_Logistic_Regression(self,D,L,DTE,l):
        v = numpy.zeros(D.shape[0] + 1)
        x = numpy.array([0, 1])
        x, f, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, v, approx_grad=True, args=(D, L, l))
        w, b = x[0:-1], x[-1]
        s = numpy.dot(numpy.array(w).T, DTE) + b

        # my_pred = []
        # correct = 0
        # for p in s:
        #     if p > 0:
        #         my_pred.append(1)
        #     else:
        #         my_pred.append(0)
        #
        # for i in range(0, len(LTE)):
        #     if LTE[i] == my_pred[i]:
        #         correct += 1
        #
        # accuracy = correct / len(LTE)
        # err = (1 - accuracy)*100
        # # print(my_pred)
        # # print(LTE)
        # print("ACCURACY: ", accuracy*100, "Lambda: ", l)
        # print("ERROR: ", err, "%")
        return s

    def logreg_obj_weighted(self, v, DTR, LTR, l, pi):
        M = DTR.shape[0]
        Z = LTR * 2.0 - 1.0

        def logreg_obj(v):
            w = vcol(v[0:M])
            b = v[-1]
            reg = 0.5 * l * numpy.linalg.norm(w) ** 2
            s = (numpy.dot(w.T, DTR) + b).ravel()
            nt = DTR[:, LTR == 0].shape[1]
            avg_risk_0 = (numpy.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
            avg_risk_1 = (numpy.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
            return reg + (pi / nt) * avg_risk_1 + (1 - pi) / (DTR.shape[1] - nt) * avg_risk_0

        return logreg_obj(v)

    def calibration_score_weighted_LR(self, D, L, DTE, l, pi):
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj_weighted, numpy.zeros(D.shape[0] + 1), approx_grad=True, args=(D,L,l,pi))
        _w = _v[0:D.shape[0]]
        _b = _v[-1]
        calibration = numpy.log(pi / (1 - pi))
        STE = numpy.dot(_w.T, DTE) + _b - calibration
        return STE, _w, _b

    def compute_scores_param(self,scores, labels, l, pi):
        scores = vrow(scores)
        _, _w,_b = self.calibration_score_weighted_LR(scores, labels, scores, l, pi)
        return _w, _b