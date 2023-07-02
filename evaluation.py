import numpy

from dimensionality_reduction import DimensionalityReduction
from gaussian_classifier import MultivariateGaussianClassifier
from logistic_regression import LogisticRegression
from SVM import SupportVectorMachine
from mlFunc import *
class Evaluation:
    def __init__(self):
        self.MVG = MultivariateGaussianClassifier()
        self.LR = LogisticRegression()
        self.svm = SupportVectorMachine()




    def MVG_evaluation(self, DTR, LTR, DTE, LTE, DP, DPE):
        k =5
        Dtr = numpy.split(DTR.T, k, axis = 1)
        Ltr = numpy.split(LTR, k)
        llrMVG = []
        labelMVG =[]

        for i in range(k):
            Dte = Dtr[i]
            print(Dte)
            Lte = Ltr[i]
            D = []
            L = []
            if i == 0:
                D.append(numpy.hstack(Dtr[i + 1:]))
                L.append(numpy.hstack(Ltr[i + 1:]))
            elif i == k - 1:
                D.append(numpy.hstack(Dtr[:i]))
                L.append(numpy.hstack(Ltr[:i]))
            else:
                D.append(numpy.hstack(Dtr[:i]))
                D.append(numpy.hstack(Dtr[i + 1:]))
                L.append(numpy.hstack(Ltr[:i]))
                L.append(numpy.hstack(Ltr[i + 1:]))

            D = numpy.hstack(D)
            L = numpy.hstack(L)






            print("---------------MVG WITHOUT LDA--------------------------")
            # s = self.MVG.predict_MVG(DTE.T, LTE)
            # DFC = evaluation(s,LTE, 0.5, 1, 10)

            self.MVG.setup_MVG(numpy.array(D), numpy.array(L))
            llrMVG = numpy.append(llrMVG,self.MVG.predict_MVG(Dte,Lte))
            llrMVG = numpy.hstack(llrMVG)
            labelMVG = numpy.append(labelMVG,Lte,axis = 0)
            labelMVG = numpy.hstack(labelMVG)

        minDFC = compute_min_DCF(numpy.array(llrMVG),numpy.array(labelMVG), 0.5, 1, 10 )
        print("MIN DFC", minDFC)





        #########################################################
        #                     DFC on test data
        #########################################################
        s = self.MVG.predict_MVG(DTE.T, LTE)
        res= compute_act_DCF(s, LTE, 0.5,1, 10, None)
        print("ACT DFC", res)
        # print("---------------MVG WITH LDA--------------------------")
        # self.MVG.setup_MVG(DP, LTR)
        # s1 = self.MVG.predict_MVG(DPE, LTE)
        # #DFC1 = evaluation(s1, LTE, 0.5, 1, 10)


        #
        # print("---------------MVG NAIVE BAYES WITHOUT LDA--------------------------")
        # self.MVG.setup_MVG_Naive_Bayes(DTR.T, LTR)
        # s2 = self.MVG.predict_MVG_Naive_Bayes(DTE.T, LTE)
        # DFC2 = evaluation(s2, LTE, 0.5, 1, 10)
        # print(DFC2)
        # print("---------------MVG NAIVE BAYES WITH LDA--------------------------")
        #
        # self.MVG.setup_MVG_Naive_Bayes(DP, LTR)
        # s3 = self.MVG.predict_MVG_Naive_Bayes(DPE, LTE)
        # DFC3 = evaluation(s3, LTE, 0.5, 1, 10)
        # print(DFC3)
        # print("---------------MVG TIED COV WITHOUT LDA--------------------------")
        #
        # self.MVG.setup_MVG_Tied_Cov(DTR.T, LTR)
        # s4 = self.MVG.predict_MVG_Tied_Cov(DTE.T, LTE)
        # DFC4 = evaluation(s4, LTE, 0.5, 1, 10)
        #
        # print(DFC4)
        # print("---------------MVG TIED COV WITH LDA--------------------------")
        #
        # self.MVG.setup_MVG_Tied_Cov(DP, LTR)
        # self.MVG.predict_MVG_Tied_Cov(DPE, LTE)
        #
        # print("---------------MVG TIED COV + NAIVE WITHOUT LDA--------------------------")
        #
        # self.MVG.setup_MVG_Tied_Cov_Naive(DTR.T, LTR)
        # self.MVG.predict_MVG_Tied_Cov_Naive(DTE.T, LTE)
        #
        # print("---------------MVG TIED COV + NAIVE WITH LDA--------------------------")
        # self.MVG.setup_MVG_Tied_Cov_Naive(DP, LTR)
        # self.MVG.predict_MVG_Tied_Cov_Naive(DPE, LTE)
        #
        # print("---------------LOGISTIC REGRESSION WITHOUT LDA--------------------------")
        # self.LR.setup_Logistic_Regression(DTR.T, LTR, 0.1)
        # self.LR.preditc_Logistic_Regression(DTE.T, LTE, 0.1)
        #
        # print("---------------LOGISTIC REGRESSION WITH LDA--------------------------")
        # self.LR.setup_Logistic_Regression(DP, LTR, 0.1)
        # self.LR.preditc_Logistic_Regression(DPE, LTE, 0.1)

        print("---------------SVM Linear REGRESSION WITHOUT LDA--------------------------")
        K = [1.0]
        C = [0.01]
        self.svm.validation_SVM(DTR.T, LTR.T, K, C, "svm")
        self.svm.evaluation_SVM(DTR.T, LTR.T, DTE.T, LTE.T, K, C, "svm_ev")
        #self.svmLin.setup_primal_svm(DTR.T, LTR, 0.1)
        #self.svmLin.predict_primal_svm(DTE.T, LTE, 0.1)
        #K_arr = [1.0]
        #C_arr = [0.01]
        #self.svmLin.validation_SVM(DTR, LTR, K_arr, C_arr, 'RAW_')
        
        #print("---------------SVM Kernel Poly REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelPoly_svm(DTR.T, LTR, DTE.T, LTE)

        #print("---------------SVM Kernel RBG REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelRBF_svm(DTR.T, LTR, DTE.T, LTE)

