import numpy

from dimensionality_reduction import DimensionalityReduction
from gaussian_classifier import MultivariateGaussianClassifier
from logistic_regression import LogisticRegression
from svm import SupportVectorMachine
from mlFunc import *
class Validation:
    def __init__(self):
        self.MVG = MultivariateGaussianClassifier()
        self.LR = LogisticRegression()
        self.svmLin = SupportVectorMachine()
    def k_fold_MVG(self, k,DTR, LTR):
        llrMVG = []
        llrNV = []
        llrTCV =[]
        llrTNV =[]
        labelMVG = []

        Dtr = numpy.split(DTR.T, k, axis=1)
        Ltr = numpy.split(LTR, k)

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
            self._getScoresMVG(Dte, D, L, llrMVG, llrNV, llrTCV, llrTNV)
            labelMVG = numpy.append(labelMVG, Lte, axis=0)

        return llrMVG,llrNV,llrTCV,llrTNV,labelMVG

    def _getScoresMVG(self, Dte, D, L, llrMVG, llrNV,llrTCV, llrTNV):
        llrs = self.MVG.predict_MVG(D, L, Dte)
        llrsNV = self.MVG.predict_MVG_Naive_Bayes(D,L,Dte)
        llrsTCV  = self.MVG.predict_MVG_Tied_Cov(D,L,Dte)
        llrsTNV = self.MVG.predict_MVG_Tied_Cov_Naive(D,L,Dte)
        llrMVG.append(llrs)
        llrNV.append(llrsNV)
        llrTCV.append(llrsTCV)
        llrTNV.append(llrsTNV)


    def MVG_validation(self, DTR, LTR, pi, C_fn, C_fp):

        llrMVG, llrNV, llrTCV,llrTNV, labelMVG = self.k_fold_MVG(5,DTR,LTR)

        minDCF_MVG = compute_min_DCF(numpy.hstack(llrMVG),numpy.hstack(labelMVG), pi, C_fn, C_fp)
        s_MVG = self.MVG.predict_MVG_Naive_Bayes(DTR.T, LTR, DTR.T)
        actDCF_MVG = compute_act_DCF(s_MVG, LTR, pi, C_fn, C_fp)
        print("############MVG###############")
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_MVG)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_MVG)
        bayes_error_min_act_plot(s_MVG,LTR, 1)

        print("############NAIVE BAYES#############")
        minDCF_NV = compute_min_DCF(numpy.hstack(llrNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        s_NV = self.MVG.predict_MVG_Naive_Bayes(DTR.T, LTR, DTR.T)
        actDCF_NV = compute_act_DCF(s_NV, LTR, pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_NV)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_NV)
        bayes_error_min_act_plot(s_NV, LTR, 1)

        print("############TIED COV#############")
        minDCF_TCV = compute_min_DCF(numpy.hstack(llrTCV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        s_TCV = self.MVG.predict_MVG_Tied_Cov(DTR.T, LTR, DTR.T)
        actDCF_TCV = compute_act_DCF(s_TCV, LTR, pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_TCV)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_TCV)

        print("############TIED COV BAYES#############")
        minDCF_TNV = compute_min_DCF(numpy.hstack(llrTNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        s_TNV = self.MVG.predict_MVG_Tied_Cov_Naive(DTR.T, LTR, DTR.T)
        actDCF_TNV = compute_act_DCF(s_TNV, LTR, pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_TNV)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_TNV)

    def LR_validation(self, k, DTR, LTR):



        return 0
        #########################################################
        #                     BOH
        #########################################################


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
        #self.svmLin.setup_primal_svm(DTR.T, LTR, 0.1)
        #self.svmLin.predict_primal_svm(DTE.T, LTE, 0.1)

        print("---------------SVM Kernel Poly REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelPoly_svm(DTR.T, LTR, DTE.T, LTE)

        print("---------------SVM Kernel RBG REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelRBF_svm(DTR.T, LTR, DTE.T, LTE)

