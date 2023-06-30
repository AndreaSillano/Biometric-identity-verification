from dimensionality_reduction import DimensionalityReduction
from gaussian_classifier import MultivariateGaussianClassifier
from logistic_regression import LogisticRegression
from svm import SupportVectorMachine

class Evaluation:
    def __init__(self):
        self.MVG = MultivariateGaussianClassifier()
        self.LR = LogisticRegression()
        self.svmLin = SupportVectorMachine()

    def MVG_evaluation(self, DTR, LTR, DTE, LTE, DP, DPE):
        print("---------------MVG WITHOUT LDA--------------------------")

        self.MVG.setup_MVG(DTR.T, LTR)
        self.MVG.predict_MVG(DTE.T, LTE)

        print("---------------MVG WITH LDA--------------------------")
        self.MVG.setup_MVG(DP, LTR)
        self.MVG.predict_MVG(DPE, LTE)

        print("---------------MVG NAIVE BAYES WITHOUT LDA--------------------------")
        self.MVG.setup_MVG_Naive_Bayes(DTR.T, LTR)
        self.MVG.predict_MVG_Naive_Bayes(DTE.T, LTE)

        print("---------------MVG NAIVE BAYES WITH LDA--------------------------")

        self.MVG.setup_MVG_Naive_Bayes(DP, LTR)
        self.MVG.predict_MVG_Naive_Bayes(DPE, LTE)

        print("---------------MVG TIED COV WITHOUT LDA--------------------------")

        self.MVG.setup_MVG_Tied_Cov(DTR.T, LTR)
        self.MVG.predict_MVG_Tied_Cov(DTE.T, LTE)

        print("---------------MVG TIED COV WITH LDA--------------------------")

        self.MVG.setup_MVG_Tied_Cov(DP, LTR)
        self.MVG.predict_MVG_Tied_Cov(DPE, LTE)

        print("---------------MVG TIED COV + NAIVE WITHOUT LDA--------------------------")

        self.MVG.setup_MVG_Tied_Cov_Naive(DTR.T, LTR)
        self.MVG.predict_MVG_Tied_Cov_Naive(DTE.T, LTE)

        print("---------------MVG TIED COV + NAIVE WITH LDA--------------------------")
        self.MVG.setup_MVG_Tied_Cov_Naive(DP, LTR)
        self.MVG.predict_MVG_Tied_Cov_Naive(DPE, LTE)

        print("---------------LOGISTIC REGRESSION WITHOUT LDA--------------------------")
        self.LR.setup_Logistic_Regression(DTR.T, LTR, 0.1)
        self.LR.preditc_Logistic_Regression(DTE.T, LTE, 0.1)

        print("---------------LOGISTIC REGRESSION WITH LDA--------------------------")
        self.LR.setup_Logistic_Regression(DP, LTR, 0.1)
        self.LR.preditc_Logistic_Regression(DPE, LTE, 0.1)

        print("---------------SVM Linear WITHOUT LDA--------------------------")
        self.svmLin.setup_primal_svm(DTR, LTR, 0.1)
        self.svmLin.predict_primal_svm(DTE.T, LTE, 0.1)

