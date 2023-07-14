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
    def plot_DCF_PCA(self,DTR, LTR,DTE,LTE ,pi, C_fn, C_fp):
        '''Plot PCA LOG'''
        #lam = numpy.logspace(-5, 1, 30)
        lam =[1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        minDCF_9 = []
        minDCF_8 = []
        minDCF_7 = []
        minDCF_LR = []
        DP_9 = self.dimRed.PCA(DTR.T, 9)
        DPE_9 = self.dimRed.PCA_DTE(DTR.T,9,DTE.T)
        DP_8 = self.dimRed.PCA(DTR.T, 8)
        DPE_8 = self.dimRed.PCA_DTE(DTR.T, 8, DTE.T)
        DP_7 = self.dimRed.PCA(DTR.T, 7)
        DPE_7 = self.dimRed.PCA_DTE(DTR.T, 7, DTE.T)
        for l in lam:
            lr1=self.LR.predict_Logistic_Regression_weigthed(DTR, LTR, DTE, l, 0.5)

            minDCF_LR = numpy.hstack(
                (minDCF_LR, compute_min_DCF(numpy.hstack(lr1), numpy.hstack(LTE), pi, C_fn, C_fp)))

            lr9 = self.LR.predict_Logistic_Regression_weigthed(DP_9, LTR, DPE_9, l, 0.5)

            minDCF_9 = numpy.hstack(
                (minDCF_9, compute_min_DCF(numpy.hstack(lr9), numpy.hstack(LTE), pi, C_fn, C_fp)))

            lr8 = self.LR.predict_Logistic_Regression_weigthed(DP_8, LTR, DPE_8, l, 0.5)

            minDCF_8 = numpy.hstack(
                (minDCF_8, compute_min_DCF(numpy.hstack(lr8), numpy.hstack(LTE), pi, C_fn, C_fp)))

            lr7 = self.LR.predict_Logistic_Regression_weigthed(DP_7, LTR, DPE_7, l, 0.5)

            minDCF_7 = numpy.hstack(
                (minDCF_7, compute_min_DCF(numpy.hstack(lr7), numpy.hstack(LTE), pi, C_fn, C_fp)))

        self.PLT.plot_DCF_compare_PCA(lam, numpy.hstack(minDCF_LR), numpy.hstack(minDCF_9), numpy.hstack(minDCF_8),
                                      numpy.hstack(minDCF_7))
    def plot_DCF_lamda_prior(self, DTR, LTR, DTE, LTE, C_fn, C_fp, norm=False):
        '''Plot minDCF on different lambda and prior'''
        lam = numpy.logspace(-5, 1, 30)
        # lam = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        minDCF_LR_0_5 = []
        minDCF_LR_0_1 = []
        minDCF_LR_0_9 = []
        if norm:
            DTR, DTE = znorm(DTR, DTE)

        for l in lam:

            lr1 = self.LR.predict_Logistic_Regression_weigthed(DTR, LTR, DTE, l, 0.5)

            minDCF_LR_0_5 = numpy.hstack(
                (minDCF_LR_0_5, compute_min_DCF(numpy.hstack(lr1), numpy.hstack(LTE), 0.5, C_fn, C_fp)))
            lr2 = self.LR.predict_Logistic_Regression_weigthed(DTR, LTR, DTE, l, 0.1)

            minDCF_LR_0_1 = numpy.hstack(
                (minDCF_LR_0_1, compute_min_DCF(numpy.hstack(lr2), numpy.hstack(LTE), 0.1, C_fn, C_fp)))

            lr3 = self.LR.predict_Logistic_Regression_weigthed(DTR, LTR, DTE, l, 0.9)
            minDCF_LR_0_9 = numpy.hstack(
                (minDCF_LR_0_9, compute_min_DCF(numpy.hstack(lr3), numpy.hstack(LTE), 0.9, C_fn, C_fp)))

        self.PLT.plot_DCF_lambda(lam, numpy.hstack(minDCF_LR_0_5), numpy.hstack(minDCF_LR_0_1),
                                 numpy.hstack(minDCF_LR_0_9), 'lambda')

    def plot_minDCF_Z(self, DTR, LTR,DTE,LTE, pi, C_fn, C_fp):
        '''Plot min DCF vs minDCF with z-norm'''
        #lam = numpy.logspace(-5, 1, 30)
        lam = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        minDCF_LR = []
        minDCF_LR_Z = []

        for l in lam:
            lr1 = self.LR.predict_Logistic_Regression_weigthed(DTR, LTR, DTE, l, 0.5)

            minDCF_LR = numpy.hstack(
                (minDCF_LR, compute_min_DCF(numpy.hstack(lr1), numpy.hstack(LTE), pi, C_fn, C_fp)))

            DTR_Z,DTE_Z = znorm(DTR,DTE)
            lr2 = self.LR.predict_Logistic_Regression_weigthed(DTR_Z, LTR, DTE_Z, l, 0.5)
            minDCF_LR_Z = numpy.hstack(
                (minDCF_LR_Z, compute_min_DCF(numpy.hstack(lr2), numpy.hstack(LTE), pi, C_fn, C_fp)))

        self.PLT.plot_DCF_compare(lam, numpy.hstack(minDCF_LR), numpy.hstack(minDCF_LR_Z))

    def plot_minDCF_Z_Q(self, DTR, LTR,DTE,LTE, pi, C_fn, C_fp):
        '''Plot min DCF vs minDCF with z-norm'''
        #lam = numpy.logspace(-5, 1, 30)
        lam = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        minDCF_LR = []
        minDCF_LR_Z = []
        expanded_DTR = numpy.apply_along_axis(self.vecxxT, 0, DTR)
        expanded_DTE = numpy.apply_along_axis(self.vecxxT, 0, DTE)
        phi = numpy.vstack([expanded_DTR, DTR])
        phi_DTE = numpy.vstack([expanded_DTE, DTE])
        DTR_Z, DTE_Z = znorm(DTR, DTE)
        expanded_DTR_Z = numpy.apply_along_axis(self.vecxxT, 0, DTR_Z)
        expanded_DTE_Z = numpy.apply_along_axis(self.vecxxT, 0, DTE_Z)
        phi_Z = numpy.vstack([expanded_DTR_Z, DTR_Z])
        phi_DTE_Z = numpy.vstack([expanded_DTE_Z, DTE_Z])


        for l in lam:
            lr1 = self.LR.predict_quad_Logistic_Regression(phi, LTR, phi_DTE, l, pi)

            minDCF_LR = numpy.hstack(
                (minDCF_LR, compute_min_DCF(numpy.hstack(lr1), numpy.hstack(LTE), pi, C_fn, C_fp)))


            lr2 = self.LR.predict_quad_Logistic_Regression(phi_Z, LTR, phi_DTE_Z, l, pi)
            minDCF_LR_Z = numpy.hstack(
                (minDCF_LR_Z, compute_min_DCF(numpy.hstack(lr2), numpy.hstack(LTE), pi, C_fn, C_fp)))

        self.PLT.plot_DCF_compare_QUAD(lam, numpy.hstack(minDCF_LR), numpy.hstack(minDCF_LR_Z))


    def plot_DCF_PCA_Q(self,DTR, LTR,DTE,LTE ,pi, C_fn, C_fp):
        '''Plot PCA LOG'''
        #lam = numpy.logspace(-5, 1, 15)
        lam =[1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        minDCF_9 = []
        minDCF_8 = []
        minDCF_7 = []
        minDCF_6 = []
        minDCF_LR = []
        expanded_DTR = numpy.apply_along_axis(self.vecxxT, 0, DTR)
        expanded_DTE = numpy.apply_along_axis(self.vecxxT, 0, DTE)
        phi = numpy.vstack([expanded_DTR, DTR])
        phi_DTE = numpy.vstack([expanded_DTE, DTE])

        DP_9 = self.dimRed.PCA(DTR.T, 9)
        DPE_9 = self.dimRed.PCA_DTE(DTR.T,9,DTE.T)
        expanded_DTR_9 = numpy.apply_along_axis(self.vecxxT, 0, DP_9)
        expanded_DTE_9 = numpy.apply_along_axis(self.vecxxT, 0, DPE_9)
        phi_9 = numpy.vstack([expanded_DTR_9, DP_9])
        phi_DTE_9 = numpy.vstack([expanded_DTE_9, DPE_9])

        DP_8 = self.dimRed.PCA(DTR.T, 8)
        DPE_8 = self.dimRed.PCA_DTE(DTR.T, 8, DTE.T)
        expanded_DTR_8 = numpy.apply_along_axis(self.vecxxT, 0, DP_8)
        expanded_DTE_8 = numpy.apply_along_axis(self.vecxxT, 0, DPE_8)
        phi_8 = numpy.vstack([expanded_DTR_8, DP_8])
        phi_DTE_8 = numpy.vstack([expanded_DTE_8, DPE_8])

        DP_7 = self.dimRed.PCA(DTR.T, 7)
        DPE_7 = self.dimRed.PCA_DTE(DTR.T, 7, DTE.T)
        expanded_DTR_7 = numpy.apply_along_axis(self.vecxxT, 0, DP_7)
        expanded_DTE_7 = numpy.apply_along_axis(self.vecxxT, 0, DPE_7)
        phi_7 = numpy.vstack([expanded_DTR_7, DP_7])
        phi_DTE_7 = numpy.vstack([expanded_DTE_7, DPE_7])

        DP_6 = self.dimRed.PCA(DTR.T, 6)
        DPE_6 = self.dimRed.PCA_DTE(DTR.T, 6, DTE.T)
        expanded_DTR_6 = numpy.apply_along_axis(self.vecxxT, 0, DP_6)
        expanded_DTE_6 = numpy.apply_along_axis(self.vecxxT, 0, DPE_6)
        phi_6 = numpy.vstack([expanded_DTR_6, DP_6])
        phi_DTE_6 = numpy.vstack([expanded_DTE_6, DPE_6])
        for l in lam:
            lr1=self.LR.predict_quad_Logistic_Regression(phi, LTR, phi_DTE, l, pi)

            minDCF_LR = numpy.hstack(
                (minDCF_LR, compute_min_DCF(numpy.hstack(lr1), numpy.hstack(LTE), pi, C_fn, C_fp)))

            lr9 = self.LR.predict_quad_Logistic_Regression(phi_9, LTR, phi_DTE_9, l, pi)

            minDCF_9 = numpy.hstack(
                (minDCF_9, compute_min_DCF(numpy.hstack(lr9), numpy.hstack(LTE), pi, C_fn, C_fp)))

            lr8 = self.LR.predict_quad_Logistic_Regression(phi_8, LTR, phi_DTE_8, l, pi)

            minDCF_8 = numpy.hstack(
                (minDCF_8, compute_min_DCF(numpy.hstack(lr8), numpy.hstack(LTE), pi, C_fn, C_fp)))

            lr7 = self.LR.predict_quad_Logistic_Regression(phi_7, LTR, phi_DTE_7, l, pi)

            minDCF_7 = numpy.hstack(
                (minDCF_7, compute_min_DCF(numpy.hstack(lr7), numpy.hstack(LTE), pi, C_fn, C_fp)))

            lr6 = self.LR.predict_quad_Logistic_Regression(phi_6, LTR, phi_DTE_6, l, pi)

            minDCF_6 = numpy.hstack(
                (minDCF_6, compute_min_DCF(numpy.hstack(lr6), numpy.hstack(LTE), pi, C_fn, C_fp)))

        self.PLT.plot_DCF_compare_PCA_Q(lam, numpy.hstack(minDCF_LR), numpy.hstack(minDCF_9), numpy.hstack(minDCF_8),
                                        numpy.hstack(minDCF_7), numpy.hstack(minDCF_6))

    def LR_evaluation(self, DTE, LTE, DTR, LTR, pi, C_fn, C_fp, norm=False):
        DTE = DTE.T
        DTR = DTR.T
        if norm:
            DTR, DTE = znorm(DTR, DTE)
        expanded_DTR = numpy.apply_along_axis(self.vecxxT, 0, DTR)
        expanded_DTE = numpy.apply_along_axis(self.vecxxT, 0, DTE)
        phi = numpy.vstack([expanded_DTR, DTR])

        phi_DTE = numpy.vstack([expanded_DTE, DTE])

        lrQ = self.LR.predict_quad_Logistic_Regression(phi, LTR, phi_DTE, 1e-4, pi)
        lr =  self.LR.predict_Logistic_Regression_weigthed(DTR, LTR, DTE, 0.1, pi)

        print("############LOGISTIC REGRESSION#############")
        minDCF_LR = compute_min_DCF(numpy.hstack(lr), numpy.hstack(LTE), pi, C_fn, C_fp)
        actDCF_LR = compute_act_DCF(numpy.hstack(lr), numpy.hstack(LTE), pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_LR)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_LR)

        print("############LOGISTIC REGRESSION QUADRATIC#############")
        minDCF_LRQ = compute_min_DCF(numpy.hstack(lrQ), numpy.hstack(LTE), pi, C_fn, C_fp)
        actDCF_LRQ = compute_act_DCF(numpy.hstack(lrQ), numpy.hstack(LTE), pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_LRQ)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_LRQ)

        #self.plot_DCF_lamda_prior(DTR, LTR,DTE,LTE, C_fn,C_fp)
        #self.plot_DCF_lamda_prior(DTR, LTR,DTE,LTE, C_fn,C_fp, True)
        #self.plot_minDCF_Z(DTR,LTR,DTE,LTE,pi,C_fn,C_fp)
        #self.plot_DCF_PCA(DTR,LTR,DTE,LTE,pi,C_fn,C_fp)
        #self.plot_minDCF_Z_Q(DTR,LTR,DTE,LTE,pi,C_fn,C_fp)
        #self.plot_DCF_PCA_Q(DTR, LTR, DTE, LTE, pi, C_fn, C_fp)

    def plot_GMM_full(self, DTR, LTR, DTE, LTE,pi, a, p, Cfn, Cfp):
        data = {
            "Non-target K = 1": [],
            "Non-target K = 2": [],
            "Non-target K = 4": [],
            "Non-target K = 8": [],
            "Non-target K = 16": [],
            "Non-target K = 32": []
        }
        for i in range(0, 6):
            # TARGET 1
            llr_GMM_Full = self.GMM.predict_GMM_full(DTR, LTR, DTE, 1, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 2
            llr_GMM_Full=self.GMM.predict_GMM_full(DTR, LTR, DTE, 2, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 4
            llr_GMM_Full = self.GMM.predict_GMM_full(DTR, LTR, DTE, 4, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))

        self.PLT.plot_bar_GMM(data)

    def plot_GMM_naive(self, DTR, LTR, DTE, LTE,pi, a, p, Cfn, Cfp):
        data = {
            "Non-target K = 1": [],
            "Non-target K = 2": [],
            "Non-target K = 4": [],
            "Non-target K = 8": [],
            "Non-target K = 16": [],
            "Non-target K = 32": []
        }
        for i in range(0, 6):
            # TARGET 1
            llr_GMM_Full = self.GMM.predict_GMM_naive(DTR, LTR, DTE, 1, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 2
            llr_GMM_Full=self.GMM.predict_GMM_naive(DTR, LTR, DTE, 2, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 4
            llr_GMM_Full = self.GMM.predict_GMM_naive(DTR, LTR, DTE, 4, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))

        self.PLT.plot_bar_GMM(data)
    def plot_GMM_tied(self, DTR, LTR, DTE, LTE,pi, a, p, Cfn, Cfp):
        data = {
            "Non-target K = 1": [],
            "Non-target K = 2": [],
            "Non-target K = 4": [],
            "Non-target K = 8": [],
            "Non-target K = 16": [],
            "Non-target K = 32": []
        }
        for i in range(0, 6):
            # TARGET 1
            llr_GMM_Full = self.GMM.predict_GMM_TiedCov(DTR, LTR, DTE, 1, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 2
            llr_GMM_Full=self.GMM.predict_GMM_TiedCov(DTR, LTR, DTE, 2, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 4
            llr_GMM_Full = self.GMM.predict_GMM_TiedCov(DTR, LTR, DTE, 4, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))

        self.PLT.plot_bar_GMM(data)

    def plot_GMM_tiedNaive(self, DTR, LTR, DTE, LTE,pi, a, p, Cfn, Cfp):
        data = {
            "Non-target K = 1": [],
            "Non-target K = 2": [],
            "Non-target K = 4": [],
            "Non-target K = 8": [],
            "Non-target K = 16": [],
            "Non-target K = 32": []
        }
        for i in range(0, 6):
            # TARGET 1
            llr_GMM_Full = self.GMM.predict_GMM_TiedNaive(DTR, LTR, DTE, 1, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 2
            llr_GMM_Full=self.GMM.predict_GMM_TiedNaive(DTR, LTR, DTE, 2, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))
            # TARGET 4
            llr_GMM_Full = self.GMM.predict_GMM_TiedNaive(DTR, LTR, DTE, 4, 2**i, a, p)
            llr = numpy.hstack(llr_GMM_Full)
            scores_tot = compute_min_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp)
            data["Non-target K = " + str(2 ** i)].append(round(scores_tot, 3))

        self.PLT.plot_bar_GMM(data)

    def get_scores_SVM(self, D, L, Dte, C, K, gamma, scoresRBF_append, pi):
        scoresRBF_append.append(self.svm.predict_SVM_RBF(D, L, C, K, Dte, gamma, False, pi))

    def SVM_evaluation(self, DTE, LTE, DTR, LTR, pi, C_fn, C_fp):
        scores_RBF = []
        labelRBF = []


        self.get_scores_SVM(DTR, LTR, DTE, 10, 0.1, 0.001, scores_RBF, pi)
        labelRBF = numpy.append(labelRBF, LTE, axis=0)
        minDCF_MVG = compute_min_DCF(numpy.hstack(scores_RBF), numpy.hstack(labelRBF), pi, C_fn, C_fp)
        actDCF_MVG = compute_act_DCF(numpy.hstack(scores_RBF), numpy.hstack(labelRBF), pi, C_fn, C_fp)
        print("############MVG###############")
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_MVG)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_MVG)
        #bayes_error_min_act_plot(numpy.hstack(llrMVG), numpy.hstack(labelMVG), 1)

    def GMM_evaluation(self, DTE, LTE, DTR, LTR, pi, Cfn, Cfp, comp, compNT, a, p):
        DTR = DTR.T
        DTE = DTE.T
        #labelGMM = numpy.append(labelGMM, Lte, axis=0)
        llr_GMM_Full = self.GMM.predict_GMM_full(DTR, LTR, DTE, comp, compNT, a, p)
        llr_GMM_Naive = self.GMM.predict_GMM_naive(DTR, LTR, DTE, comp, compNT, a, p)
        llr_GMM_Tied = self.GMM.predict_GMM_TiedCov(DTR, LTR, DTE, comp, compNT, a, p)
        llr_GMM_TiedNaive = self.GMM.predict_GMM_TiedNaive(DTR, LTR, DTE, comp, compNT, a, p)


        # print("##########GMM FULL############")
        # llr = numpy.hstack(llr_GMM_Full)
        # scores_tot = compute_min_DCF(llr, LTE, pi, Cfn, Cfp)
        # print(f'- components  %1i | with prior = {pi} -> minDCF = %.3f ' % (comp, scores_tot))
        # rettt = compute_act_DCF(llr, numpy.hstack(LTE), pi, Cfn, Cfp, None)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % rettt)

        # print("##########GMM NAIVE##########")
        # llrN = numpy.hstack(llr_GMM_Naive)
        # scores_totN = compute_min_DCF(llrN, LTE, pi, Cfn, Cfp)
        # print(f'- components  %1i | with prior = {pi} -> minDCF = %.3f ' % (comp, scores_totN))
        # rettt = compute_act_DCF(llrN, numpy.hstack(LTE), pi, Cfn, Cfp, None)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % rettt)

        # print("##########GMM TIED##########")
        # llrT = numpy.hstack(llr_GMM_Tied)
        # scores_totT = compute_min_DCF(llrT, LTE, pi, Cfn, Cfp)
        # print(f'- components  %1i | with prior = {pi} -> minDCF = %.3f ' % (comp, scores_totT))
        # rettt = compute_act_DCF(llrT, numpy.hstack(LTE), pi, Cfn, Cfp, None)
        # print(f'- with prior = {pi} -> actDCF = %.3f' % rettt)

        print("##########GMM TIED NAIVE##########")
        llrTN = numpy.hstack(llr_GMM_TiedNaive)
        scores_totTN = compute_min_DCF(llrTN, LTE, pi, Cfn, Cfp)
        print(f'- components  %1i | with prior = {pi} -> minDCF = %.3f ' % (comp, scores_totTN))
        rettt = compute_act_DCF(llrTN, numpy.hstack(LTE), pi, Cfn, Cfp, None)
        print(f'- with prior = {pi} -> actDCF = %.3f' % rettt)

        #self.plot_GMM_full(DTR,LTR,DTE,LTE, pi,a,p,Cfn,Cfp)
        #self.plot_GMM_naive(DTR,LTR,DTE,LTE, pi,a,p,Cfn,Cfp)
        #self.plot_GMM_tied(DTR,LTR,DTE,LTE, pi,a,p,Cfn,Cfp)
        #self.plot_GMM_tiedNaive(DTR,LTR,DTE,LTE, pi,a,p,Cfn,Cfp)

    def plot_minDCF_cal_score(self, DTR, LTR, DTE,LTE, pi):
        # MVG
        DTR = DTR.T
        DTE = DTE.T
        DP_8 = self.dimRed.PCA(DTR.T, 8)
        DPE_8 = self.dimRed.PCA_DTE(DTR.T, 8, DTE.T)
        llrMVG = self.MVG.predict_MVG(DP_8, LTR, DPE_8)
        bayes_error_min_act_plot(numpy.hstack(llrMVG), LTE, 1)

        # qlog

        DP_7 = self.dimRed.PCA(DTR.T, 7)
        DPE_7 = self.dimRed.PCA_DTE(DTR.T, 7,DTE.T)


        expanded_DTR = numpy.apply_along_axis(self.vecxxT, 0, DP_7)
        expanded_DTE = numpy.apply_along_axis(self.vecxxT, 0, DPE_7)
        phi = numpy.vstack([expanded_DTR, DP_7])

        phi_DTE = numpy.vstack([expanded_DTE, DPE_7])

        lrQ = self.LR.predict_quad_Logistic_Regression(phi, LTR, phi_DTE, 1e-4, pi)

        bayes_error_min_act_plot(numpy.hstack(lrQ), LTE, 1)
        _w, _b = self.LR.compute_scores_param(numpy.hstack(lrQ), LTE, 0.001, 0.7)

        # cal_score = numpy.dot(_w.T,numpy.hstack(lrQ).reshape(1, numpy.hstack(lrQ).shape[0])) #- numpy.log(pi/(1-pi))
        cal_score_lr = _w * lrQ + _b - numpy.log(pi / (1 - pi))
        bayes_error_min_act_plot(numpy.hstack(cal_score_lr), LTE, 1)
        # svm

        scoresRBF_append = self.svm.predict_SVM_RBF(DTR, LTR, 0.1, 10, DTE, 0.001, False, pi)

        bayes_error_min_act_plot(numpy.hstack(scoresRBF_append), LTE, 1)
        _w, _b = self.LR.compute_scores_param(numpy.hstack(scoresRBF_append), LTE, 0.001, 0.7)
        cal_score_RBF = _w * scoresRBF_append + _b - numpy.log(pi / (1 - pi))
        bayes_error_min_act_plot(numpy.hstack(cal_score_RBF), LTE, 1)
        # gmm
        # _, llr_GMM_Naive, _, _, llr_GMM_labels = self.kfold_GMM(5, DTR, LTR, 1, 8, 0.1, 0.01)
        DP_71 = self.dimRed.PCA(DTR.T, 7)
        DPE_71 = self.dimRed.PCA_DTE(DTR.T, 7, DTE.T)
        llr_GMM_Naive = self.GMM.predict_GMM_naive(DP_71, LTR, DPE_71, 1, 16, 0.1, 0.01)


        bayes_error_min_act_plot(numpy.hstack(llr_GMM_Naive), LTE, 1)

        bayes_error_min_act_plot_compare(numpy.hstack(llrMVG), numpy.hstack(cal_score_lr), numpy.hstack(cal_score_RBF),
                                         numpy.hstack(llr_GMM_Naive), LTE, LTE, LTE, LTE, 1)

