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