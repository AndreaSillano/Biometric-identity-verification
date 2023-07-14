import numpy

from dimensionality_reduction import DimensionalityReduction
from gaussian_classifier import MultivariateGaussianClassifier
from logistic_regression import LogisticRegression
from svm import SupportVectorMachine
from dimensionality_reduction import DimensionalityReduction
from GMM import GMM
from mlFunc import *
from plotter import Plotter

class Evaluator:
    def __init__(self):
        self.MVG = MultivariateGaussianClassifier()
        self.LR = LogisticRegression()
        self.svm = SupportVectorMachine()
        self.GMM = GMM()
        self.PLT = Plotter()
        self.dimRed = DimensionalityReduction()

    def _getScoresMVG(self, Dte, D, L, llrMVG, llrNV,llrTCV, llrTNV):
        llrs = self.MVG.predict_MVG(D, L, Dte)
        llrsNV = self.MVG.predict_MVG_Naive_Bayes(D,L,Dte)
        llrsTCV  = self.MVG.predict_MVG_Tied_Cov(D,L,Dte)
        llrsTNV = self.MVG.predict_MVG_Tied_Cov_Naive(D,L,Dte)
        llrMVG.append(llrs)
        llrNV.append(llrsNV)
        llrTCV.append(llrsTCV)
        llrTNV.append(llrsTNV)

    def MVG_evaluation(self, DTE, LTE, DTR, LTR, pi, C_fn, C_fp):
        DTE = DTE.T
        DTR = DTR.T
        llrMVG = []
        llrNV = []
        llrTCV = []
        llrTNV = []
        labelMVG = []


        self._getScoresMVG(DTE, DTR, LTR, llrMVG, llrNV, llrTCV, llrTNV)
        labelMVG = numpy.append(labelMVG, LTE, axis=0)
        minDCF_MVG = compute_min_DCF(numpy.hstack(llrMVG), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        actDCF_MVG = compute_act_DCF(numpy.hstack(llrMVG), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        print("############MVG###############")
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_MVG)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_MVG)
        #bayes_error_min_act_plot(numpy.hstack(llrMVG), numpy.hstack(labelMVG), 1)


        # minDCF_MVG = compute_min_DCF(numpy.hstack(llrMVG), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        # actDCF_MVG = compute_act_DCF(numpy.hstack(llrMVG), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        # print("############MVG###############")
        # print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_MVG)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_MVG)

        # # bayes_error_min_act_plot(numpy.hstack(llrMVG),numpy.hstack(labelMVG), 1)
        # 
        # print("############NAIVE BAYES#############")
        # minDCF_NV = compute_min_DCF(numpy.hstack(llrNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        # actDCF_NV = compute_act_DCF(numpy.hstack(llrNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        # print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_NV)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_NV)
        # bayes_error_min_act_plot(numpy.hstack(llrNV),LTR, 1)
        # 
        # print("############TIED COV#############")
        # minDCF_TCV = compute_min_DCF(numpy.hstack(llrTCV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        # actDCF_TCV = compute_act_DCF(numpy.hstack(llrTCV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        # print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_TCV)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_TCV)
        # # bayes_error_min_act_plot(numpy.hstack(llrTCV), numpy.hstack(labelMVG), 1)

        print("############TIED COV BAYES#############")
        minDCF_TNV = compute_min_DCF(numpy.hstack(llrTNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        actDCF_TNV = compute_act_DCF(numpy.hstack(llrTNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_TNV)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_TNV)
        # #bayes_error_min_act_plot(numpy.hstack(llrMVG), numpy.hstack(labelMVG), 1)
    def vecxxT(self, x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        return xxT

    def LR_evaluation(self, DTE, LTE, DTR, LTR, pi, C_fn, C_fp):
        labelLr =[]
        expanded_DTR = numpy.apply_along_axis(self.vecxxT, 0, DTR)
        expanded_DTE = numpy.apply_along_axis(self.vecxxT, 0, DTE)
        phi = numpy.vstack([expanded_DTR, DTR])

        phi_DTE = numpy.vstack([expanded_DTE, DTE])

        labelLr = numpy.append(labelLr, LTE, axis=0)
        lrQ = self.LR.predict_quad_Logistic_Regression(phi, LTR, phi_DTE, 0.01, pi)

        print("############LOGISTIC REGRESSION QUADRATIC#############")
        minDCF_LRQ = compute_min_DCF(numpy.hstack(lrQ), numpy.hstack(LTE), pi, C_fn, C_fp)
        actDCF_LRQ = compute_act_DCF(numpy.hstack(lrQ), numpy.hstack(LTE), pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_LRQ)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_LRQ)

        #bayes_error_min_act_plot(numpy.hstack(lrQ), numpy.hstack(labelLr), 1)

    def get_scores_SVM(self, D, L, Dte, scoresLin_append, scoresPol_append, scoresRBF_append, pi):    
        scoresLin_append.append(self.svm.predict_SVM_Linear(D, L, 10, 1, Dte, False, pi))
        #scoresPol_append.append(self.svm.predict_SVM_Pol(D, L, 0.1, 10, Dte, 0, 2, False, pi))
        #scoresRBF_append.append(self.svm.predict_SVM_RBF(D, L, 10, 0.1, Dte, 1e-3, False, pi))

    def SVM_evaluation(self, DTE, LTE, DTR, LTR, pi, C_fn, C_fp):
        scores_Lin = []
        scores_Pol = []
        scores_RBF = []
        label = []
        C_arr = numpy.logspace(-5, 1, 15)

        self.get_scores_SVM(DTR, LTR, DTE, scores_Lin, scores_Pol, scores_RBF, pi)
        label = numpy.append(label, LTE, axis=0)

        minDCF_Lin = compute_min_DCF(numpy.hstack(scores_Lin), numpy.hstack(label), pi, C_fn, C_fp)
        # actDCF_Lin = compute_act_DCF(numpy.hstack(scores_Lin), numpy.hstack(label), pi, C_fn, C_fp)
        # print("############Lin###############")
        # print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_Lin)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_Lin)
        # #bayes_error_min_act_plot(numpy.hstack(llrMVG), numpy.hstack(labelMVG), 1)

        # minDCF_Pol = compute_min_DCF(numpy.hstack(scores_Pol), numpy.hstack(label), pi, C_fn, C_fp)
        # actDCF_Pol = compute_act_DCF(numpy.hstack(scores_Pol), numpy.hstack(label), pi, C_fn, C_fp)
        # print("############Pol###############")
        # print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_Pol)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_Pol)
        # #bayes_error_min_act_plot(numpy.hstack(llrMVG), numpy.hstack(labelMVG), 1)

        # minDCF_RBF = compute_min_DCF(numpy.hstack(scores_RBF), numpy.hstack(label), pi, C_fn, C_fp)
        # actDCF_RBF = compute_act_DCF(numpy.hstack(scores_RBF), numpy.hstack(label), pi, C_fn, C_fp)
        # print("############RBF###############")
        # print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_RBF)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_RBF)
        # #bayes_error_min_act_plot(numpy.hstack(llrMVG), numpy.hstack(labelMVG), 1)
        self.get_scores_SVM(self.dimRed(DTR.T, 7), LTR, self.dimRed(DTE.T, 7), scores_Pol, 0, scores_RBF, pi)
        minDCF_Pol = compute_min_DCF(numpy.hstack(scores_Pol), numpy.hstack(label), pi, C_fn, C_fp)
        D, Dte = znorm(DTR, DTE)
        self.get_scores_SVM(D, LTR, Dte, scores_RBF, 0, 0, pi)
        minDCF_RBF = compute_min_DCF(numpy.hstack(scores_RBF), numpy.hstack(label), pi, C_fn, C_fp)
        self.PLT.plot_DCF_SVM_C(C_arr, numpy.hstack(minDCF_Lin), numpy.hstack(minDCF_Pol), numpy.hstack(minDCF_RBF), 'C', 'comp')

    def GMM_evaluation(self, DTE, LTE, DTR, LTR, pi, Cfn, Cfp, comp, compNT, a, p):

        #labelGMM = numpy.append(labelGMM, Lte, axis=0)
        llr_GMM_Full = self.GMM.predict_GMM_full(DTR, LTR, DTE, comp, compNT, a, p)

        print("##########GMM FULL##########")
        llr = numpy.hstack(llr_GMM_Full)
        scores_tot = compute_min_DCF(llr, LTE, pi, Cfn, Cfp)
        print(f'- components  %1i | with prior = {pi} -> minDCF = %.3f ' % (comp, scores_tot))
        #rettt = compute_act_DCF(llr, llr_GMM_labels, pi, Cfn, Cfp, None)
        #print(f'- with prior = {pi} -> actDCF = %.3f' % rettt)

