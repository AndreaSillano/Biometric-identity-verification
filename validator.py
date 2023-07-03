import numpy

from dimensionality_reduction import DimensionalityReduction
from gaussian_classifier import MultivariateGaussianClassifier
from logistic_regression import LogisticRegression
from SVM import SupportVectorMachine
from mlFunc import *
class Validation:
    def __init__(self):
        self.MVG = MultivariateGaussianClassifier()
        self.LR = LogisticRegression()
        self.svmLin = SupportVectorMachine()

    def _getScores(self, Dte, D, L, llrMVG, llrNV):
        llrs = self.MVG.predict_MVG(D, L, Dte)
        llrsNV = self.MVG.predict_MVG_Naive_Bayes(D,L,Dte)
        llrMVG.append(llrs)
        llrNV.append(llrsNV)
        return llrMVG, llrNV


    def MVG_validation(self, DTR, LTR, DTE, LTE):
        k = 5
        Dtr = numpy.split(DTR.T, k, axis = 1)
        Ltr = numpy.split(LTR, k)
        #Ltep = numpy.split(DPE, k)

        llrMVG = []
        llrNV = []
        labelMVG =[]




        for i in range(k):
            Dte = Dtr[i]
            Lte = Ltr[i]
            D = []
            L = []

            for j in range(k):
                if j != i:
                    D.append(Dtr[j])
                    L.append(Ltr[j])

            D = numpy.hstack(D)
            L = numpy.hstack(L)

            # Train the model

            print("---------------MVG WITHOUT LDA--------------------------")

            self._getScores(Dte,D,L,llrMVG,llrNV)
            labelMVG = numpy.append(labelMVG,Lte,axis = 0)

        minDFC = compute_min_DCF(numpy.hstack(llrMVG),numpy.hstack(labelMVG), 0.5, 1, 10)
        print("MIN DFC", minDFC)
        minDFC = compute_min_DCF(numpy.hstack(llrNV), numpy.hstack(labelMVG), 0.5, 1, 10)
        print("MIN DFC", minDFC)





        #########################################################
        #                     DFC on test data
        #########################################################

        stt = self.MVG.predict_MVG(DTR.T, LTR, DTR.T)
        rettt = compute_act_DCF(numpy.hstack(stt), LTR, 0.5, 1, 10, None)
        print("ACT DFC ON TRAIN", rettt)



        s1 = self.MVG.predict_MVG_Naive_Bayes(DTR.T, LTR, DTR.T)
        res1= compute_act_DCF(numpy.hstack(s1), LTR, 0.5,1, 10, None)
        print("ACT DFC TRAIN BAYES", res1)
        # print("---------------MVG WITH LDA--------------------------")
        # self.MVG.setup_MVG(DP, LTR)
        # s1 = self.MVG.predict_MVG(DPE, LTE)
        # #DFC1 = evaluation(s1, LTE, 0.5, 1, 10)
        #bayes_error_min_act_plot(DTE.T, LTE, 2)

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
        self.svmLin.validation_SVM(DTR.T, LTR, [0.1], [1], "validation svm")
        self.svmLin.evaluation_SVM(DTR.T, LTR, DTE.T, LTE, [0.1], [1], "ev svm")
        #self.svmLin.predict_primal_svm(DTE.T, LTE, 0.1)

        #print("---------------SVM Kernel Poly REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelPoly_svm(DTR.T, LTR, DTE.T, LTE)

        #print("---------------SVM Kernel RBG REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelRBF_svm(DTR.T, LTR, DTE.T, LTE)

