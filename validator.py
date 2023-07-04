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
        self.svm = SupportVectorMachine()
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


    def MVG_validation(self, DTR, LTR, pi, C_fn, C_fp, DTE, LTE):

        llrs = self.MVG.predict_MVG(DTR.T, LTR, DTE.T)
        minDCF_MVG_test = compute_min_DCF(llrs,LTE, pi, C_fn, C_fp)
        #s_MVG = self.MVG.predict_MVG(DTR.T,LTR , DTR.T)
        actDCF_MVG_test = compute_act_DCF(llrs,LTE, pi, C_fn, C_fp)
        #actDCF_MVG = compute_act_DCF(s_MVG, LTR, pi, C_fn, C_fp)
        print("############MVG###############")
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_MVG_test)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_MVG_test)


        llrMVG, llrNV, llrTCV,llrTNV, labelMVG = self.k_fold_MVG(5,DTR,LTR)

        minDCF_MVG = compute_min_DCF(numpy.hstack(llrMVG),numpy.hstack(labelMVG), pi, C_fn, C_fp)
        #s_MVG = self.MVG.predict_MVG(DTR.T,LTR , DTR.T)
        actDCF_MVG = compute_act_DCF(numpy.hstack(llrMVG), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        #actDCF_MVG = compute_act_DCF(s_MVG, LTR, pi, C_fn, C_fp)
        print("############MVG###############")
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_MVG)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_MVG)
        bayes_error_min_act_plot(numpy.hstack(llrMVG),numpy.hstack(labelMVG), 1)

        print("############NAIVE BAYES#############")
        minDCF_NV = compute_min_DCF(numpy.hstack(llrNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        actDCF_NV = compute_act_DCF(numpy.hstack(llrNV), numpy.hstack(labelMVG),pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_NV)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_NV)
        bayes_error_min_act_plot(numpy.hstack(llrNV),LTR, 1)

        print("############TIED COV#############")
        minDCF_TCV = compute_min_DCF(numpy.hstack(llrTCV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        actDCF_TCV = compute_act_DCF(numpy.hstack(llrTCV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_TCV)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_TCV)
        bayes_error_min_act_plot(numpy.hstack(llrTCV), numpy.hstack(labelMVG), 1)

        print("############TIED COV BAYES#############")
        minDCF_TNV = compute_min_DCF(numpy.hstack(llrTNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        s_TNV = self.MVG.predict_MVG_Tied_Cov_Naive(DTR.T, LTR, DTR.T)
        actDCF_TNV = compute_act_DCF(numpy.hstack(llrTNV), numpy.hstack(labelMVG), pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_TNV)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_TNV)
        bayes_error_min_act_plot(numpy.hstack(llrTNV), LTR, 1)

    def k_fold_LR(self,k,DTR,LTR):
        lr_score = []
        labelLR = []
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
            labelLR = numpy.append(labelLR, Lte, axis=0)
            lr_score.append(self.LR.preditc_Logistic_Regression(D, L, Dte, 0.00001))


        return lr_score, labelLR

    def LR_validation(self,DTR, LTR, pi, C_fn, C_fp):
        lr, labelLr = self.k_fold_LR(5,DTR,LTR)
        print("############LOGISTIC REGRESSION#############")
        minDCF_LR = compute_min_DCF(numpy.hstack(lr), numpy.hstack(labelLr), pi, C_fn, C_fp)
        actDCF_LR = compute_act_DCF(numpy.hstack(lr), numpy.hstack(labelLr),pi, C_fn, C_fp)
        print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF_LR)
        print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF_LR)

        #bayes_error_min_act_plot(s_LR, LTR, 1)

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

        #print("---------------SVM Linear REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.validation_SVM(DTR.T, LTR, [0.1], [1], "validation svm")
        #self.svmLin.evaluation_SVM(DTR.T, LTR, DTE.T, LTE, [0.1], [1], "ev svm")
        #self.svmLin.predict_primal_svm(DTE.T, LTE, 0.1)

        #print("---------------SVM Kernel Poly REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelPoly_svm(DTR.T, LTR, DTE.T, LTE)

        #print("---------------SVM Kernel RBG REGRESSION WITHOUT LDA--------------------------")
        #self.svmLin.setup_kernelRBF_svm(DTR.T, LTR, DTE.T, LTE)
    def kfold_SVM(self, DTR, LTR, K, C):
        k = 5
        Dtr = numpy.split(DTR, k, axis=1)
        Ltr = numpy.split(LTR, k)

        scoresLin_append = []
        scoresPol_append = []
        scoresRBF_append = []
        PCA_SVM_scoresLin_append = []
        PCA2_SVM_scoresLin_append = []
        SVM_labels = []

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

            Dte = Dtr[i]
            Lte = Ltr[i]

            wStar, primal = self.svm.train_SVM_linear(D, L, C, K)
            DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])

            scores = numpy.dot(wStar.T, DTEEXT).ravel()
            scoresLin_append.append(scores)

            costant = 0
            degree = 2
            aStar, primal = self.svm.train_SVM_polynomial(D, L, C, K, costant, degree)
            Z = numpy.zeros(L.shape)
            Z[L == 1] = 1
            Z[L == 0] = -1
            kernel = (numpy.dot(D.T, Dte) + costant) ** degree + K * K
            scores = numpy.sum(numpy.dot(aStar * vrow(Z), kernel), axis=0)
            scoresPol_append.append(scores)

            Z = L * 2 - 1
            gamma=1.
            aStar, loss = self.svm.train_SVM_RBF(D, L, C, K, gamma)
            kern = numpy.zeros((D.shape[1], Dte.shape[1]))
            for i in range(D.shape[1]):
                for j in range(Dte.shape[1]):
                    kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
            scores = numpy.sum(numpy.dot(aStar * vrow(Z), kern), axis=0)
            scoresRBF_append.append(scores)

            SVM_labels = numpy.append(SVM_labels, Lte, axis=0)
            SVM_labels = numpy.hstack(SVM_labels)

        return scoresLin_append, scoresPol_append, scoresRBF_append, SVM_labels

        # plot_ROC(scoresLin_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

        # Cfn and Ctp are set to 1
        #bayes_error_min_act_plot(scoresLin_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)

        ###############################

        # π = 0.1
        #scores_tot = compute_min_DCF(scoresLin_append, SVM_labels, 0.1, 1, 1)
        #print(scores_tot)
        ###############################

        # π = 0.9
        #scores_tot = compute_min_DCF(scoresLin_append, SVM_labels, 0.9, 1, 1)
        #print(scores_tot)
    def kfold_calibration_SVM(self, DTR, LTR, K, C):
        k = 5
        Dtr = numpy.split(DTR, k, axis=1)
        Ltr = numpy.split(LTR, k)
        scoresLin_append = []
        scoresPol_append = []
        scoresRBF_append = []
        PCA_SVM_scoresLin_append = []
        PCA2_SVM_scoresLin_append = []
        SVM_labels = []

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

            Dte = Dtr[i]
            Lte = Ltr[i]


            #
            # wStar, primal = self.svm.train_SVM_linear(D, L, C, K)
            # DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])
            #
            # scores = numpy.dot(wStar.T, DTEEXT).ravel()
            # scoresLin_append.append(scores)
            #
            # costant = 0
            # degree = 2
            # aStar, primal = self.svm.train_SVM_polynomial(D, L, C, K, costant, degree)
            # Z = numpy.zeros(L.shape)
            # Z[L == 1] = 1
            # Z[L == 0] = -1
            # kernel = (numpy.dot(D.T, Dte) + costant) ** degree + K * K
            # scores = numpy.sum(numpy.dot(aStar * vrow(Z), kernel), axis=0)
            # scoresPol_append.append(scores)

            Z = L * 2 - 1
            gamma=1.
            aStar, loss = self.svm.train_SVM_RBF(D, L, C, K, gamma)
            kern = numpy.zeros((D.shape[1], D.shape[1]))
            for i in range(D.shape[1]):
                for j in range(D.shape[1]):
                    kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
            scoresT = numpy.sum(numpy.dot(aStar * vrow(Z), kern), axis=0)
            SVM_labels = numpy.append(SVM_labels, L, axis=0)
            SVM_labels = numpy.hstack(SVM_labels)

            a,b = self.LR.compute_scores_param(scoresT, L, 1e-4 ,0.5)

            Z = L * 2 - 1
            gamma = 1.
            aStar, loss = self.svm.train_SVM_RBF(D, L, C, K, gamma)
            kern = numpy.zeros((D.shape[1], Dte.shape[1]))
            for i in range(D.shape[1]):
                for j in range(Dte.shape[1]):
                    kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
            scoresV = numpy.sum(numpy.dot(aStar * vrow(Z), kern), axis=0)
            computeLLR = a * scoresV + b - numpy.log(0.5 / (1 - 0.5))

            SVM_labels = numpy.append(SVM_labels, Lte, axis=0)
            SVM_labels = numpy.hstack(SVM_labels)


            scoresRBF_append.append(computeLLR)


        return scoresRBF_append, SVM_labels

    def SVM_validation(self, DTR, LTR, DTE, LTE):
        K = 1
        scoresLin_append = []
        scoresPol_append = []
        scoresRBF_append = []
        SVM_labels = []
        DTR = DTR.T
        scoresLin_append, scoresPol_append, scoresRBF_append, SVM_labels = self.kfold_SVM(DTR, LTR, K, 0.1)

        # scoresLin_append = numpy.hstack(scoresLin_append)
        # scores_tot = compute_min_DCF(scoresLin_append, SVM_labels, 0.5, 1, 10)
        # print("MIN DFC TRAIN SVM Linear", scores_tot)
        #
        # wStar, primal = self.svm.train_SVM_linear(DTR, LTR, 0.1, K) #testerai anche su DTR
        # DTEEXT = numpy.vstack([DTR, K * numpy.ones((1, DTR.shape[1]))])
        # scores = numpy.dot(wStar.T, DTEEXT).ravel()
        # scoresLin_append = []
        # scoresLin_append.append(scores)
        # rettt = compute_act_DCF(numpy.hstack(scoresLin_append), LTR, 0.5, 1, 10, None)
        # print("ACT DFC ON TRAIN SVM Linear", rettt)
        #
        # scoresPol_append = numpy.hstack(scoresPol_append)
        # scores_tot = compute_min_DCF(scoresPol_append, SVM_labels, 0.5, 1, 10)
        # print("MIN DFC TRAIN SVM Polynomial", scores_tot)
        #
        # costant = 0
        # degree = 2
        # aStar, primal = self.svm.train_SVM_polynomial(DTR, LTR, 0.1, K)
        # Z = numpy.zeros(LTR.shape)
        # Z[LTR == 1] = 1
        # Z[LTR == 0] = -1
        # kernel = (numpy.dot(DTR.T, DTR) + costant) ** degree + K * K
        # scores = numpy.sum(numpy.dot(aStar * vrow(Z), kernel), axis=0)
        # scoresPol_append = []
        # scoresPol_append.append(scores)
        # rettt = compute_act_DCF(numpy.hstack(scoresPol_append), LTR, 0.5, 1, 10, None)
        # print("ACT DFC ON TRAIN SVM Polynomial", rettt)

        scoresRBF_append = numpy.hstack(scoresRBF_append)
        scores_tot = compute_min_DCF(scoresRBF_append, SVM_labels, 0.5, 1, 10)
        print("MIN DFC TRAIN SVM RBF", scores_tot)

        rettt = compute_act_DCF(numpy.hstack(scoresRBF_append), LTR, 0.5, 1, 10, None)
        print("ACT DFC ON TRAIN SVM RBF", rettt)

        cal_score, cal_label = self.kfold_calibration_SVM(DTR, LTR, K, 0.1)
        rettt = compute_act_DCF(numpy.hstack(cal_score), cal_label, 0.5, 1, 10, None)
        print("ACT DFC ON TRAIN SVM RBF - CAL", rettt)

