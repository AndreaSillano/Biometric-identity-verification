import numpy
import scipy
from scipy import optimize

class LogisticRegression:
    def __init__(self):
        self.x= numpy.array([0,1])
        self.f={}
        self.d ={}
        self.w = {}
        self.b ={}
        self.v = {}
    def setup_Logistic_Regression(self, D, LTR ,l):
        self.v = numpy.zeros(D.shape[0] + 1)

        self.x, self.f, self.d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, self.v, approx_grad=True, args=(D, LTR, l))
        self.w, self.b = self.x[0:-1], self.x[-1]
        #print("w: ", x[0:-1], "b: ", x[-1])

    def logreg_obj(self,v, DTR, LTR, l):
        w, b = v[0:-1], v[-1]
        zi = 2 * LTR - 1
        f = (l / 2) * numpy.sum(w ** 2) + (1 / DTR.shape[1]) * numpy.sum(
            numpy.logaddexp(0, -zi * (numpy.dot(w.T, DTR) + b)))
        return (f)

    def preditc_Logistic_Regression(self,DTE,LTE,l):
        s = numpy.dot(self.w.T, DTE) + self.b
        #print(s)
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
        print("ACCURACY: ", accuracy*100, "Lambda: ", l)
        print("ERROR: ", err, "%")